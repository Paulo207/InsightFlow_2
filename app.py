import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ======================================================
# Configurações globais
# ======================================================
st.set_page_config(
    page_title="Análise Exploratória e Financeira Avançada",
    layout="wide",
    initial_sidebar_state="expanded"
)

PLOT_TEMPLATE = "seaborn"

# ======================================================
# Cache de carregamento
# ======================================================
@st.cache_data(show_spinner="Carregando dados...")
def load_data(file, sep=",", encoding="utf-8"):
    if file.name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    return pd.read_csv(file, sep=sep, encoding=encoding, low_memory=False)

# ======================================================
# Normalização segura de tipos
# ======================================================
def normalizar_tipos_seguro(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        s = df[col].replace(["", "nan", "NaN", "None", None], np.nan)

        # tenta converter para número
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().any():
            df[col] = num
            continue

        # tenta converter para data
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if dt.notna().any():
            df[col] = dt
            continue

        # fallback para string
        df[col] = s.astype(str)

    df = df.dropna(axis=1, how="all")
    return df

# ======================================================
# 🔒 Arrow-safe (ESSENCIAL)
# ======================================================
def arrow_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")

        else:
            df[col] = (
                df[col]
                .astype(str)
                .replace({"nan": None, "NaT": None, "None": None})
            )

    return df

# ======================================================
# Detecção automática de colunas financeiras
# ======================================================
def detectar_colunas_financeiras(df: pd.DataFrame):
    palavras = [
        "valor", "preço", "preco", "custo", "total",
        "mensal", "anual", "faturamento", "receita"
    ]

    return [
        col for col in df.select_dtypes(include="number").columns
        if any(p in col.lower() for p in palavras)
    ]

# ======================================================
# App principal
# ======================================================
def main():
    st.title("📊 Análise Exploratória e Financeira Avançada")
    st.markdown("Upload de **CSV ou Excel** → análises automáticas e gráficas seguras.")

    # ---------------- Upload ----------------
    st.sidebar.header("1️⃣ Upload do Arquivo")
    arquivo = st.sidebar.file_uploader(
        "Selecione um arquivo",
        type=["csv", "xlsx", "xls"]
    )

    if not arquivo:
        st.info("⬅️ Faça o upload para começar.")
        st.stop()

    sep = st.sidebar.selectbox("Separador (CSV)", [",", ";", "|", "\t"])
    encoding = st.sidebar.selectbox(
        "Encoding",
        ["utf-8", "latin-1", "iso-8859-1", "cp1252"]
    )

    try:
        df_raw = load_data(arquivo, sep, encoding)
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        st.stop()

    with st.spinner("Normalizando dados..."):
        df = normalizar_tipos_seguro(df_raw)
        df = arrow_safe_df(df)

    st.success(f"✅ {df.shape[0]:,} linhas × {df.shape[1]} colunas")

    # ---------------- Seleção ----------------
    st.sidebar.header("2️⃣ Seleção de Colunas")
    cols = st.sidebar.multiselect(
        "Colunas",
        df.columns.tolist(),
        default=df.columns.tolist()
    )
    df = df[cols]

    # ---------------- Filtros ----------------
    st.sidebar.header("3️⃣ Filtros")
    with st.sidebar.form("filtros"):
        df_filtrado = df.copy()

        num_cols = df_filtrado.select_dtypes(include="number").columns.tolist()
        cat_cols = df_filtrado.select_dtypes(exclude="number").columns.tolist()

        for col in cat_cols:
            valores = sorted(df_filtrado[col].dropna().astype(str).unique())
            if 1 < len(valores) <= 50:
                sel = st.multiselect(col, valores, default=valores)
                df_filtrado = df_filtrado[df_filtrado[col].astype(str).isin(sel)]

        for col in num_cols:
            min_v, max_v = df_filtrado[col].min(), df_filtrado[col].max()
            if pd.notna(min_v) and pd.notna(max_v) and min_v != max_v:
                intervalo = st.slider(
                    col,
                    float(min_v),
                    float(max_v),
                    (float(min_v), float(max_v))
                )
                df_filtrado = df_filtrado[df_filtrado[col].between(*intervalo)]

        st.form_submit_button("Aplicar filtros")

    df_filtrado = arrow_safe_df(df_filtrado)

    # ---------------- Visualização ----------------
    st.subheader("🔍 Dados Filtrados")
    st.dataframe(df_filtrado.head(200), width="stretch")

    st.subheader("ℹ️ Informações")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Tipos de dados**")
        st.write(df_filtrado.dtypes.astype(str))
    with c2:
        st.write("**Nulos (%)**")
        st.write((df_filtrado.isna().mean() * 100).round(2))

    # ---------------- Financeiro ----------------
    col_fin = detectar_colunas_financeiras(df_filtrado)

    if col_fin:
        st.subheader("💰 Análise Financeira")
        col = st.selectbox("Coluna financeira", col_fin)
        valores = df_filtrado[col].dropna()

        if not valores.empty:
            st.columns(4)[0].metric("Total", f"R$ {valores.sum():,.2f}")
            st.columns(4)[1].metric("Média", f"R$ {valores.mean():,.2f}")
            st.columns(4)[2].metric("Mediana", f"R$ {valores.median():,.2f}")
            st.columns(4)[3].metric("Registros", len(valores))

            top10 = df_filtrado.nlargest(10, col)

            tabs = st.tabs(["🍕 Pizza", "📊 Barra", "🌳 Treemap"])
            with tabs[0]:
                st.plotly_chart(
                    px.pie(top10, values=col, names=top10.index, template=PLOT_TEMPLATE),
                    use_container_width=True
                )
            with tabs[1]:
                st.plotly_chart(
                    px.bar(top10, x=col, y=top10.index, orientation="h", template=PLOT_TEMPLATE),
                    use_container_width=True
                )
            with tabs[2]:
                st.plotly_chart(
                    px.treemap(top10, path=[top10.index], values=col, template=PLOT_TEMPLATE),
                    use_container_width=True
                )

    # ---------------- Exploração Numérica ----------------
    if num_cols:
        st.subheader("📈 Exploração Visual")
        tabs = st.tabs(["Histograma", "Boxplot", "Dispersão", "Correlação"])

        with tabs[0]:
            c = st.selectbox("Variável", num_cols, key="hist")
            st.plotly_chart(px.histogram(df_filtrado, x=c, template=PLOT_TEMPLATE), use_container_width=True)

        with tabs[1]:
            c = st.selectbox("Variável", num_cols, key="box")
            st.plotly_chart(px.box(df_filtrado, y=c, template=PLOT_TEMPLATE), use_container_width=True)

        with tabs[2]:
            if len(num_cols) >= 2:
                x = st.selectbox("X", num_cols, key="sx")
                y = st.selectbox("Y", num_cols, index=1, key="sy")
                st.plotly_chart(px.scatter(df_filtrado, x=x, y=y, template=PLOT_TEMPLATE), use_container_width=True)

        with tabs[3]:
            if len(num_cols) >= 2:
                corr = df_filtrado[num_cols].corr()
                st.plotly_chart(
                    px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r"),
                    use_container_width=True
                )

    # ---------------- Exportação ----------------
    st.subheader("⬇️ Exportar")
    df_export = arrow_safe_df(df_filtrado)
    csv = df_export.to_csv(index=False, encoding="utf-8-sig").encode()

    st.download_button(
        "Baixar CSV",
        data=csv,
        file_name="dados_filtrados.csv",
        mime="text/csv"
    )

# ======================================================
if __name__ == "__main__":
    main()
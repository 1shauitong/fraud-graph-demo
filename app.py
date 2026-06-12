import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile
import os
import requests
import json
import matplotlib.pyplot as plt

# ===================== 全局配置 =====================
st.set_page_config(
    page_title="反诈资金图谱演示系统",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- DeepSeek 大模型配置区（仅需修改这里） ----------------------
DEEPSEEK_API_KEY = "sk-89852ded89b943379d319f6711f6948c"  # 替换为你的真实API密钥
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL_NAME = "deepseek-v4-flash"  # 可选：deepseek-coder
# =====================================================================

# ===================== 工具函数：DeepSeek 大模型调用 =====================
def call_deepseek_llm(prompt: str) -> str:
    """调用DeepSeek在线大模型API，兼容OpenAI格式"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": DEEPSEEK_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 2000
    }
    try:
        resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=body, timeout=30)
        res_json = resp.json()
        return res_json["choices"][0]["message"]["content"]
    except Exception as e:
        return f"大模型调用失败：{str(e)}，请检查API密钥是否正确、网络是否正常"

# ===================== 加载离线数据（缓存加速） =====================
@st.cache_data
def load_all_data():
    data_path = "./data/"
    node_df = pd.read_csv(data_path + "node_processed.csv", dtype={"账户ID": str})
    edge_df = pd.read_csv(data_path + "test_edge.csv", dtype={"付款账户ID": str, "收款账户ID": str})
    risk_df = pd.read_csv(data_path + "risk_score_table.csv", dtype={"账户ID": str})
    top20_df = pd.read_csv(data_path + "top20_accounts.csv", dtype={"账户ID": str})
    case_df = pd.read_csv(data_path + "typical_cases.csv")
    return node_df, edge_df, risk_df, top20_df, case_df

node_df, edge_df, risk_df, top20_df, case_df = load_all_data()

# ===================== 侧边栏导航 =====================
with st.sidebar:
    st.title("🔍 反诈资金图谱系统")
    st.divider()
    page = st.radio(
        "功能导航",
        [
            "项目总览 & 指标",
            "涉诈账户风险识别",
            "交互式资金图谱",
            "可疑链路挖掘",
            "AI智能研判报告"
        ]
    )
    st.divider()
    st.info(f"当前大模型：DeepSeek {DEEPSEEK_MODEL_NAME}")

# ===================== 页面1：项目总览 & 指标 =====================
if page == "项目总览 & 指标":
    st.title("基于资金图谱的涉诈账户发现与可疑链路解释")
    st.subheader("第五届中国研究生金融科技创新大赛 · 江苏银行赛题")
    st.divider()

    # 核心指标卡片
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("模型AUC", "0.892", "↑ 较基线+7.9%")
    with col2:
        st.metric("PR-AUC", "0.875", "↑ 较基线+24.6%")
    with col3:
        st.metric("Top5%风险覆盖率", "58.2%", "≥50% 达标")
    with col4:
        st.metric("Top20关联命中率", "62.5%", "≥50% 达标")

    st.divider()
    st.subheader("项目整体工作流")
    if os.path.exists("./static/workflow.png"):
        st.image("./static/workflow.png", use_column_width=True)
    else:
        st.warning("未找到工作流图片，请将 workflow.png 放入 static 文件夹")

    st.subheader("模型效果对比")
    compare_df = pd.DataFrame({
        "模型": ["逻辑回归", "XGBoost", "GCN图神经网络"],
        "AUC": [0.735, 0.812, 0.892],
        "PR-AUC": [0.702, 0.783, 0.875]
    })
    st.bar_chart(compare_df, x="模型", y=["AUC", "PR-AUC"], use_container_width=True)

# ===================== 页面2：涉诈账户风险识别 =====================
elif page == "涉诈账户风险识别":
    st.title("📊 涉诈账户风险识别")
    st.divider()

    # 风险筛选
    risk_filter = st.selectbox("按风险等级筛选", ["全部", "高风险", "中风险", "低风险"])
    if risk_filter != "全部":
        filter_risk = risk_df[risk_df["风险等级"] == risk_filter]
    else:
        filter_risk = risk_df

    st.subheader(f"账户列表（共 {len(filter_risk)} 条）")
    st.dataframe(
        filter_risk.sort_values("风险评分", ascending=False),
        use_container_width=True,
        height=400
    )

    # 风险分布
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("风险等级分布")
        fig1, ax1 = plt.subplots()
        values = filter_risk["风险等级"].value_counts()
        colors_pie = {"低风险": "#4CAF50", "中风险": "#FF9800", "高风险": "#f44336"}
        ax1.pie(values, labels=values.index, autopct="%1.1f%%",
                colors=[colors_pie.get(l, "#999") for l in values.index],
                startangle=90)
        ax1.axis("equal")
        st.pyplot(fig1)
    with col_b:
        st.subheader("账户类型分布")
        fig2, ax2 = plt.subplots()
        values2 = node_df["账户类型"].value_counts()
        colors2 = {"对私": "#42A5F5", "对公": "#66BB6A"}
        ax2.pie(values2, labels=values2.index, autopct="%1.1f%%",
                colors=[colors2.get(l, "#999") for l in values2.index],
                startangle=90)
        ax2.axis("equal")
        st.pyplot(fig2)

# ===================== 页面3：交互式资金图谱（PyVis） =====================
elif page == "交互式资金图谱":
    st.title("🕸️ 动态资金图谱可视化")
    st.divider()

    # 选择核心账户
    high_risk_acc = risk_df[risk_df["风险等级"] == "高风险"]["账户ID"].unique().tolist()
    select_acc = st.selectbox("选择核心高风险账户", high_risk_acc)

    # 跳数扩展开关
    hop_depth = st.radio("资金流向广度", ["1跳 (直连)", "2跳 (含中转)"], horizontal=True)

    # 构建子图
    st.subheader(f"账户 {select_acc} 周边资金流向图谱")
    G = nx.DiGraph()

    # 1跳：直接交易对手
    hop1 = edge_df[(edge_df["付款账户ID"] == select_acc) | (edge_df["收款账户ID"] == select_acc)]
    # 2跳：找到1跳节点的对手
    if hop_depth == "2跳 (含中转)":
        hop1_nodes = set(hop1["付款账户ID"].unique()) | set(hop1["收款账户ID"].unique())
        hop2 = edge_df[(edge_df["付款账户ID"].isin(hop1_nodes)) | (edge_df["收款账户ID"].isin(hop1_nodes))]
        sub_edge = pd.concat([hop1, hop2]).drop_duplicates()
    else:
        sub_edge = hop1

    if len(sub_edge) == 0:
        st.warning(f"⚠️ 账户 {select_acc} 在当前数据中没有发现交易记录（属于孤立节点），请选择其他账户查看")
    else:
        for _, row in sub_edge.iterrows():
            G.add_edge(
                row["付款账户ID"],
                row["收款账户ID"],
                time=row["交易时间分桶"],
                amount=row["金额分箱"]
            )

        # 如果G没有节点，手动添加核心节点以便展示
        if G.number_of_nodes() == 0:
            G.add_node(select_acc)

        # PyVis 可视化配置（浅色背景更清晰）
        net = Network(height="650px", width="100%", bgcolor="#ffffff", font_color="#333333", directed=True)
        net.from_nx(G)

        # 节点配色
        for node in net.nodes:
            nid = node["id"]
            if nid == select_acc:
                node["color"] = "#ff4444"
                node["size"] = 30
                node["title"] = "🛑 核心涉诈账户"
                node["font"] = {"color": "#ff4444", "size": 18, "face": "bold"}
            elif nid in high_risk_acc:
                node["color"] = "#ff8800"
                node["size"] = 18
                node["title"] = "⚠️ 高风险账户"
            else:
                node["color"] = "#4488ff"
                node["size"] = 14
                node["title"] = "交易对手"

        # 边标签
        for edge in net.edges:
            edge["color"] = "#888888"
            edge["width"] = 2

        # 生成临时HTML并展示
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            net.write_html(tmp_file.name)
            html_path = tmp_file.name

        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        components.html(html_content, height=680, scrolling=True)
        os.unlink(html_path)

        with st.expander("📊 图谱统计"):
            st.write(f"- 节点数：{G.number_of_nodes()}")
            st.write(f"- 边数：{G.number_of_edges()}")
            st.write(f"- 1跳交易笔数：{len(hop1)}")

    # 大模型解读图谱（仅当有边数据时）
    st.divider()
    st.subheader("🤖 DeepSeek AI 智能解读资金链路")
    if len(sub_edge) > 0:
        if st.button("AI 分析该账户交易特征"):
            prompt = f"""
            结合银行反诈场景，分析账户{select_acc}的资金交易图谱特征：
            1. 交易边数量：{len(sub_edge)}
            2. 请识别是否存在分散入账、集中转出、多层转账等涉诈典型模式
            3. 给出反诈风控建议，语言简洁、贴合银行业务
            """
            with st.spinner("DeepSeek AI 分析中..."):
                ai_result = call_deepseek_llm(prompt)
            st.success("AI分析完成：")
            st.write(ai_result)
    else:
        st.info("该账户无交易数据，无法进行AI链路分析")

# ===================== 页面4：可疑链路挖掘 =====================
elif page == "可疑链路挖掘":
    st.title("⛓️ 可疑资金链路 & 团伙挖掘")
    st.divider()

    st.subheader("Top20 高关联可疑账户")
    st.dataframe(top20_df, use_container_width=True)

    st.divider()
    st.subheader("典型涉诈案例 & 多跳链路")
    case_id = st.selectbox("选择案例", case_df["案例ID"].unique())
    case_info = case_df[case_df["案例ID"] == case_id].iloc[0]

    st.markdown(f"**案例描述**：{case_info['案件描述']}")
    st.markdown(f"**资金链路**：{case_info['资金链路']}")
    st.markdown(f"**证据链**：{case_info['证据链']}")
    st.markdown(f"**人工研判结论**：{case_info['研判结论']}")

    # 大模型补充分析
    if st.button("DeepSeek AI 深度分析该链路风险"):
        prompt = f"""
        金融反诈场景分析以下资金链路风险：
        案件：{case_info['案件描述']}
        链路：{case_info['资金链路']}
        现有证据：{case_info['证据链']}
        要求：补充风险点、识别诈骗团伙特征、给出处置优先级。
        """
        with st.spinner("AI思考中..."):
            ai_res = call_deepseek_llm(prompt)
        st.info("AI 深度分析结果：")
        st.write(ai_res)

# ===================== 页面5：AI智能研判报告 =====================
elif page == "AI智能研判报告":
    st.title("📄 DeepSeek AI 自动生成反诈辅助研判报告")
    st.divider()

    # 选择账户生成报告
    acc_list = risk_df["账户ID"].unique().tolist()
    report_acc = st.selectbox("选择待研判账户", acc_list)

    if st.button("一键生成 AI 研判报告", type="primary"):
        # 读取账户基础信息
        acc_info = node_df[node_df["账户ID"] == report_acc].iloc[0]
        risk_info = risk_df[risk_df["账户ID"] == report_acc].iloc[0]

        # 安全获取可选字段（列缺失时用默认值）
        def safe_get(df_row, col_name, default="暂无数据"):
            return str(df_row[col_name]) if col_name in df_row.index else default

        acct_type = safe_get(acc_info, "账户类型")
        tenure = safe_get(acc_info, "开户时长分箱")
        region = safe_get(acc_info, "地区编码")
        risk_score_val = safe_get(risk_info, "风险评分")
        risk_level_val = safe_get(risk_info, "风险等级")

        # 构造大模型提示词
        prompt = f"""
        请以银行反诈运营人员身份，撰写一份标准《涉诈账户智能研判报告》，要求正式、专业、可直接用于人工复核。
        账户基础信息：
        账户ID：{report_acc}
        账户类型：{acct_type}
        开户时长分箱：{tenure}
        地区编码：{region}
        风险评分：{risk_score_val}
        风险等级：{risk_level_val}

        要求报告包含：1.账户概况 2.风险评估 3.资金链路特征 4.佐证证据 5.风控处置建议。
        """
        with st.spinner("DeepSeek AI 正在生成专业研判报告..."):
            report_content = call_deepseek_llm(prompt)


        st.divider()
        st.subheader("📋 智能研判报告（AI生成）")
        st.write(report_content)
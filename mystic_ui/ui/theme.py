import streamlit as st


def inject_global_theme() -> None:
	"""Inject Coinbase-dark inspired CSS and sensible typography/spacing tokens.

	This is safe to call multiple times; Streamlit de-dupes identical style blocks.
	"""
	css = """
	<style>
	:root {
		--accent: #1652F0;
		--bg: #0F1115;
		--bg-soft: #151A1F;
		--text: #E6E8EA;
		--text-dim: #C7CBD1;
		--success: #16C784;
		--danger: #FF5A5F;
		--warning: #F59E0B;
		--radius-sm: 8px;
		--radius-md: 12px;
		--radius-lg: 16px;
		--space-1: 6px;
		--space-2: 10px;
		--space-3: 14px;
		--space-4: 18px;
		--space-5: 24px;
		--font-size-base: 15.5px;
		--line-height-base: 1.5;
	}

	html, body, .stApp {
		background-color: var(--bg) !important;
		color: var(--text) !important;
		font-size: var(--font-size-base);
		line-height: var(--line-height-base);
	}

	/* Typography scale */
	.stMarkdown p, .stMarkdown li, .stText, .stSelectbox, .stMultiSelect, .stTextInput, .stNumberInput {
		font-size: var(--font-size-base) !important;
		line-height: var(--line-height-base) !important;
		color: var(--text) !important;
	}
	section h1, .stMarkdown h1 { font-size: 28px; font-weight: 700; letter-spacing: 0.2px; }
	section h2, .stMarkdown h2 { font-size: 22px; font-weight: 700; letter-spacing: 0.2px; }
	section h3, .stMarkdown h3 { font-size: 18px; font-weight: 600; letter-spacing: 0.2px; }

	/* Cards/containers */
	.block-container { padding-top: var(--space-5) !important; }
	.css-1r6slb0, .css-1w7i0p6, .stCard, .stMetric, .element-container > div {
		background-color: var(--bg-soft) !important;
		border-radius: var(--radius-md) !important;
		border: 1px solid rgba(255,255,255,0.06) !important;
	}
	/* Reduce inner padding for metrics/cards */
	.stMetric { padding: var(--space-3) var(--space-4) !important; }
	.stMetric label, .stMetric .metric-label { color: var(--text-dim) !important; }
	.stMetric .metric-value { font-weight: 700 !important; }

	/* Tables */
	[data-testid="stTable"] table {
		background: var(--bg-soft) !important;
		border-radius: var(--radius-md) !important;
		border-collapse: separate !important;
		border-spacing: 0 !important;
		font-size: 15px !important;
	}
	[data-testid="stTable"] thead th {
		background: #11151B !important;
		color: var(--text-dim) !important;
		font-weight: 600 !important;
		padding: 10px 12px !important;
		border-bottom: 1px solid rgba(255,255,255,0.06) !important;
	}
	[data-testid="stTable"] tbody td {
		color: var(--text) !important;
		padding: 10px 12px !important;
		border-bottom: 1px solid rgba(255,255,255,0.05) !important;
	}
	/* Row striping */
	[data-testid="stTable"] tbody tr:nth-child(even) td { background: rgba(255,255,255,0.02) !important; }
	[data-testid="stTable"] tbody tr:hover td { background: rgba(22,82,240,0.06) !important; }

	/* Chips/Badges */
	.badge, .chip, .stBadge {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 4px 8px;
		border-radius: 999px;
		background: rgba(22,82,240,0.12);
		color: #CFE0FF;
		border: 1px solid rgba(22,82,240,0.35);
		font-weight: 600;
		font-size: 12.5px;
	}

	/* Buttons */
	.stButton > button {
		background: var(--accent) !important;
		color: white !important;
		border: 1px solid rgba(22,82,240,0.65) !important;
		border-radius: var(--radius-sm) !important;
		padding: 8px 14px !important;
		font-weight: 700 !important;
		transition: transform .03s ease, box-shadow .15s ease, background .15s ease;
		box-shadow: 0 4px 16px rgba(22,82,240,0.25);
	}
	.stButton > button:hover {
		background: #2A66FF !important;
		box-shadow: 0 6px 20px rgba(22,82,240,0.35);
		transform: translateY(-0.5px);
	}
	.stButton > button:focus { outline: 2px solid rgba(22,82,240,0.6) !important; }
	.stButton > button:active { transform: translateY(0); box-shadow: 0 2px 10px rgba(22,82,240,0.25); }

	/* Tabs */
	[data-baseweb="tab-list"] { gap: 6px !important; }
	[data-baseweb="tab"] {
		background: transparent !important;
		border: 1px solid rgba(255,255,255,0.06) !important;
		border-radius: 10px !important;
		color: var(--text-dim) !important;
		padding: 8px 12px !important;
		font-weight: 600 !important;
	}
	[data-baseweb="tab"][aria-selected="true"] {
		background: rgba(22,82,240,0.15) !important;
		border-color: rgba(22,82,240,0.55) !important;
		color: #DCE7FF !important;
	}

	/* Inputs */
	.stTextInput > div > div > input,
	.stNumberInput input,
	.stSelectbox [data-baseweb="select"] input {
		background: #10141A !important;
		color: var(--text) !important;
		border-radius: var(--radius-sm) !important;
		border: 1px solid rgba(255,255,255,0.08) !important;
	}
	.stTextInput > div > div > input:focus {
		border-color: rgba(22,82,240,0.7) !important;
		box-shadow: 0 0 0 3px rgba(22,82,240,0.25) !important;
	}

	/* Selects & Multiselects */
	[data-baseweb="select"] > div {
		background: #10141A !important;
		border-radius: var(--radius-sm) !important;
		border: 1px solid rgba(255,255,255,0.08) !important;
	}
	[data-baseweb="select"] div[role="listbox"] {
		background: #0F1319 !important;
		border: 1px solid rgba(255,255,255,0.08) !important;
	}
	[data-baseweb="select"] [role="option"] {
		color: var(--text) !important;
	}
	[data-baseweb="select"] [aria-selected="true"] {
		background: rgba(22,82,240,0.15) !important;
		color: #DCE7FF !important;
	}

	/* Expander, sidebar, and misc paddings */
	.stExpander, .stSidebar, [data-testid="stSidebar"] div[role="complementary"] {
		background: var(--bg-soft) !important;
		border-radius: var(--radius-md) !important;
		border: 1px solid rgba(255,255,255,0.06) !important;
	}
	.stExpander .streamlit-expanderHeader { font-weight: 700 !important; }

	/* Code blocks */
	code, pre {
		background: #0E1218 !important;
		border-radius: 8px !important;
		border: 1px solid rgba(255,255,255,0.06) !important;
	}

	</style>
	"""
	st.markdown(css, unsafe_allow_html=True)  # type: ignore[attr-defined]




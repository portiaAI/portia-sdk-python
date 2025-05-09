import base64
import json
import os

import streamlit as st
from dotenv import load_dotenv

from portia import (
    Config,
    ExecutionAgentType,
    LLMModel,
    LLMProvider,
    Portia,
    example_tool_registry,
)
from portia.cli import CLIExecutionHooks
from portia.config import LogLevel

load_dotenv(override=True)


# Page configuration
st.set_page_config(layout="centered", page_title="Companies Checker", page_icon="icon.png")
encoded = base64.b64encode(open("backgorund.jpg", "rb").read()).decode()
st.markdown(f"""
        <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded}");
                background-size: cover;
            }}
            .stTabs [data-baseweb="tab-list"] {{
                gap: 2px;
            }}
            .stTabs [data-baseweb="tab"] {{
                height: 40px;
                border-radius: 4px 4px 4px 4px;
                gap: 5px;
                padding: 10px;
                font-size: 24px;
                color: black;
            }}
            .stSlider [data-baseweb=slider]{{
                width: 75%;
                padding: 10px 15px 5px 15px;
                color: black;
                border-radius: 4px 4px 4px 4px;
                background-color: rgb(212, 211, 210);
                border-color: rgb(151, 146, 137);
            }}
            .stButton > button:first-child {{
                background-color: rgb(212, 211, 210);
                border-color: rgb(151, 146, 137);
            }}
            .stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"]{{
                background-color: rgb(14, 38, 74);
                box-shadow: rgb(14 38 74 / 20%) 0px 0px 0px 0.2rem;
            }}
            .stSlider > div[data-baseweb="slider"] > div > div > div > div
                {{ color: rgb(14, 38, 74); }}
        </style>
    """,
    unsafe_allow_html=True
)

# Headers and Tabs
st.header("Companies Checker", divider=True)
# Description
input_value = st.text_input("What company do you want to check?", key="input_1f")

if st.button("Search", key="b_3f"):
    with st.spinner("Planning...", show_time=True):
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
        google_config = Config.from_default(
            llm_provider=LLMProvider.GOOGLE_GENERATIVE_AI,
            llm_model_name=LLMModel.GEMINI_2_0_FLASH,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.0,
            execution_agent_type=ExecutionAgentType.ONE_SHOT,
            default_log_level=LogLevel.DEBUG,
        )
        portia = Portia(
            config=google_config,
            tools=example_tool_registry,
            execution_hooks=CLIExecutionHooks(),
        )
        plan = portia.plan(
            f"""
                Retrieve a detailed and trustworthy summary about a company {input_value}, covering aspects such as its business activities, history, age, and overall legitimacy.

                Company Details:
                    - Search the web for the company's official website, business directories, and industry databases to extract the company’s overview, core activities, and sector(s) of operation.
                    - Retrieve historical information such as the founding date, key milestones, and overall age of the company.

                Legitimacy Verification:
                    - Identify and verify business registration details, any available regulatory certifications, and other relevant legitimacy indicators.
                    - Check for any red flags or legal issues mentioned in reputable sources.

                Reputation and Reviews:
                    - Look up the company’s website ratings and customer reviews on popular review platforms:
                        -- Trustpilot: Gather ratings, user comments, and overall feedback.
                        -- Glassdoor: Check employee reviews and ratings.

                Combine all Web search-requests into a single sten with-multinle. search-queries)

                Aggregation and Presentation:
                    - Overview: Brief description and core business activities.
                    - History & Age: Key historical milestones and the founding date/age of the company.
                    - Legitimacy: Verification details including registration and any regulatory or legal insights.
                    - Reputation: Summary of reviews and ratings across different platfroms like Trustpilot and Checkatrade along with notable consumer feedback.

                Cite all relevant sources to ensure traceability and reliability of the information.
                Use one LLM query to retrieve all the information and summarize it in a single step.

                Final Output:
                    The final result should be a clear, and concise report that enables the user to understand the company’s operational background, its business legitimacy, and the customer sentiment as seen on industry-leading review sites.
            """)

        st.session_state["steps_to_execute"] = [f"**{step.tool_id}**: {step.task}" for step in plan.steps]
        # Display the steps
        if "steps_to_execute" in st.session_state:
            st.markdown("**Steps to execute:**")
            for step in st.session_state["steps_to_execute"]:
                st.markdown(f"- {step}")
            st.markdown("=====================================================================================")

    with st.spinner("Executing...", show_time=True):
        output_value = list(json.loads(portia.run_plan(plan).model_dump_json())["outputs"]["step_outputs"].values())[-1]["summary"]
        st.session_state["output"] = output_value

# Display the output
if "output" in st.session_state:
    st.markdown(st.session_state["output"])

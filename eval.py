import concurrent.futures
import os
import time
import traceback
from collections import Counter

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
from portia.plan_run import PlanRunState

load_dotenv(override=True)

def run_company_check(company_name):
    """Run a company check using Portia."""
    print(f"Starting check for company: {company_name}")

    google_config = Config.from_default(
        llm_provider=LLMProvider.GOOGLE_GENERATIVE_AI,
        llm_model_name=LLMModel.GEMINI_2_0_FLASH,
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
            Retrieve a detailed and trustworthy summary about a company {company_name}, covering aspects such as its business activities, history, age, and overall legitimacy.

            Company Details:
                - Search the web for the company's official website, business directories, and industry databases to extract the company's overview, core activities, and sector(s) of operation.
                - Retrieve historical information such as the founding date, key milestones, and overall age of the company.

            Legitimacy Verification:
                - Identify and verify business registration details, any available regulatory certifications, and other relevant legitimacy indicators.
                - Check for any red flags or legal issues mentioned in reputable sources.

            Reputation and Reviews:
                - Look up the company's website ratings and customer reviews on popular review platforms:
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
                The final result should be a clear, and concise report that enables the user to understand the company's operational background, its business legitimacy, and the customer sentiment as seen on industry-leading review sites.
        """)

    print(f"[{company_name}] Created plan with {len(plan.steps)} steps")

    try:
        plan_run = portia.run_plan(plan)
        print(f"[{company_name}] Plan execution completed with state: {plan_run.state}")
        return company_name, plan_run, None
    except Exception as e:
        error_details = {
            "exception_type": type(e).__name__,
            "exception_message": str(e),
            "traceback": traceback.format_exc()
        }
        print(f"[{company_name}] Plan execution failed with exception: {error_details['exception_type']}: {error_details['exception_message']}")
        return company_name, None, error_details

def process_result(result, exceptions_dict):
    """Process the result tuple from a completed company check."""
    company_name, plan_run, exception = result

    if exception:
        print(f"❌ Run for {company_name} failed with exception: {exception['exception_type']}: {exception['exception_message']}")
        exceptions_dict[company_name] = exception
        return "exception"

    if plan_run.state == PlanRunState.COMPLETE:
        print(f"✅ Run for {company_name} completed successfully")
        return "success"
    print(f"❌ Run for {company_name} failed with state: {plan_run.state}")
    exceptions_dict[company_name] = {"exception_type": "Failed State", "exception_message": str(plan_run.outputs.final_output)}
    return "failed"

def main():
    # List of companies to check
    companies = ["Apple", "Microsoft", "Google", "Amazon", "Meta", "Tesla", "Nvidia", "Netflix", "Disney", "IBM", "Portia AI", "OpenAI", "Anthropic"]

    # Stats
    total_runs = len(companies)

    print(f"Running {total_runs} company checks in parallel...\n")

    start_time = time.time()

    # Number of parallel workers (adjust based on your system capabilities)
    max_workers = 5

    results = Counter()
    exceptions_by_company = {}

    # Run the checks in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_company = {executor.submit(run_company_check, company): company for company in companies}

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_company):
            company = future_to_company[future]
            try:
                result = future.result()
                results[process_result(result, exceptions_by_company)] += 1
            except Exception as e:
                error_details = {
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                    "traceback": traceback.format_exc()
                }
                print(f"❌ Run for {company} failed with unexpected error: {error_details['exception_type']}: {error_details['exception_message']}")
                exceptions_by_company[company] = error_details
                results["exception"] += 1

    end_time = time.time()

    # Print summary
    print("\n===== SUMMARY =====")
    print(f"Total runs: {total_runs}")
    print(f"Successful runs: {results['success']} ({results['success']/total_runs*100:.1f}%)")
    print(f"Failed runs: {results['failed']} ({results['failed']/total_runs*100:.1f}%)")
    print(f"Exception runs: {results['exception']} ({results['exception']/total_runs*100:.1f}%)")
    print(f"Total time elapsed: {end_time - start_time:.2f} seconds")

    # Print exceptions details
    if exceptions_by_company:
        print("\n===== EXCEPTIONS DETAILS =====")
        for company, error in exceptions_by_company.items():
            print(f"\n>> Company: {company}")
            print(f"   Exception Type: {error['exception_type']}")
            print(f"   Message: {error['exception_message']}")
            if "traceback" in error:
                print("\n   Traceback:")
                print("   " + "\n   ".join(error["traceback"].split("\n")))
            print("-" * 80)

if __name__ == "__main__":
    main()

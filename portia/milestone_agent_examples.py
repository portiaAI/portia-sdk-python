from portia.milestone_plan import MilestonePlanBuilder

newsletter_plan = (
    MilestonePlanBuilder()
    .milestone(
        name="gather_emails",
        task="Find all emails relating to AI in the last 24 hours and provide a detailed "
        "summary of each one. The summary should contain the top 5 news items in each email, "
        "along with key insights, data points and any links to sources.",
        allowed_tool_prefixes=["portia:google:gmail:search_email"],
    )
    .milestone(
        name="compile_newsletter",
        task="""Compile a newsletter from the emails that have been "
gathered.

The content should be organised into topics, with each topic having:
- A title
- A summary, including insights and data points drawn from the emails
- A list of links to sources

The newsletter should be in a format that is easy to read and understand, and should be
no more than 2 pages long.
""",
        allowed_tool_prefixes=[],
    )
    .milestone(
        name="send_newsletter",
        task="Send the newsletter to sam@portialabs.ai and then exit immediately.",
        allowed_tool_prefixes=["portia:google:gmail:send_email"],
    )
    .starting_milestone("gather_emails")
    .build()
)

linear_tickets_plan = (
    MilestonePlanBuilder()
    .milestone(
        name="download_doc",
        task="Download the 'Portia Evals' doc from Google Drive and extract the Work Items section",
        allowed_tool_prefixes=["portia:google:"],
    )
    .milestone(
        name="add_tickets",
        task="""Convert the bullet points in the Work Items section into Linear tickets

and add them to the Linear project `Evals and Prompts`.

NB the team ID is 0d6ebd77-2755-4bf2-a654-13e252e61ac6
""",
        allowed_tool_prefixes=["portia:mcp:mcp.linear.app:"],
    )
    .starting_milestone("download_doc")
    .build()
)

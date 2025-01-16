"""Default examples that are passed to the query planner."""

from portia.plan import Plan, PlanContext, Step, Variable

DEFAULT_EXAMPLE_PLANS: list[Plan] = [
    Plan(
        plan_context=PlanContext(
            query="Send hello@portialabs.ai an email with a summary of the latest news on AI",
            tool_ids=["search_tool", "send_email_tool", "other_tool"],
        ),
        steps=[
            Step(
                task="Find and summarize the latest news on artificial intelligence",
                tool_name="search_tool",
                output="$ai_search_results",
            ),
            Step(
                task="Email $email politely with $ai_search_results",
                inputs=[
                    Variable(
                        name="$ai_search_results",
                        description="summary of AI news",
                    ),
                    Variable(
                        name="$email",
                        value="hello@portialabs.ai",
                        description="The email address to send the email to",
                    ),
                ],
                tool_name="send_email_tool",
                output="$final_output",
            ),
        ],
    ),
    Plan(
        plan_context=PlanContext(
            query="Compare the weather of a city in the Southern hemisphere with that of a city in the Northern hemisphere. Email the results to hello@portialabs.ai.",  # noqa: E501
            tool_ids=["search_tool", "send_email_tool", "other_tool", "weather_tool"],
        ),
        steps=[
            Step(
                task="What is a city in the Southern hemisphere?",
                tool_name="search_tool",
                output="$southern_hemisphere_city",
            ),
            Step(
                task="What is a city in the Northern hemisphere?",
                tool_name="search_tool",
                output="$northern_hemisphere_city",
            ),
            Step(
                task="What is the weather in the city in the input?",
                inputs=[
                    Variable(
                        name="$southern_hemisphere_city",
                        description="City in the southern hemisphere",
                    ),
                ],
                tool_name="weather_tool",
                output="$southern_hemisphere_city_weather",
            ),
            Step(
                task="What is the weather in the city in the input?",
                inputs=[
                    Variable(
                        name="$northern_hemisphere_city",
                        description="City in the norther hemisphere",
                    ),
                ],
                tool_name="weather_tool",
                output="$northern_hemisphere_city_weather",
            ),
            Step(
                task="Compare the weather of the 2 cities ($southern_hemisphere_city_weather and $northern_hemisphere_city_weather) and write a comparison summarizing the similarities and differences",  # noqa: E501
                inputs=[
                    Variable(
                        name="$southern_hemisphere_city_weather",
                        description="Weather of a city in the southern hemisphere",
                    ),
                    Variable(
                        name="$northern_hemisphere_city_weather",
                        description="Weather of a city in the northern hemisphere",
                    ),
                ],
                output="$weather_comparison",
            ),
            Step(
                task="Email hello@portialabs.ai with a $weather_comparison",
                inputs=[
                    Variable(
                        name="$weather_comparison",
                        description="Comparison of the weather in the two cities",
                    ),
                ],
                tool_name="send_email_tool",
                output="If the email was successfully sent",
            ),
        ],
    ),
    Plan(
        plan_context=PlanContext(
            query="Send an email to hello@portialabs.ai with the weather in London",
            tool_ids=["weather_tool", "send_email_tool", "other_tool"],
        ),
        steps=[
            Step(
                task="What is the weather in London?",
                tool_name="weather_tool",
                output="$london_weather",
            ),
            Step(
                task="Email $email_address politely with $london_weather",
                inputs=[
                    Variable(
                        name="$london_weather",
                        description="Weather in London",
                    ),
                    Variable(
                        name="$email_address",
                        value="hello@portialabs.ai",
                        description="The email address",
                    ),
                ],
                tool_name="send_email_tool",
                output="If the email was successfully sent",
            ),
        ],
    ),
]

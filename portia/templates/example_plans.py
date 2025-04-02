"""Default examples that are passed to the query planning_agent if none are provided."""

from portia.plan import OutputReference, Plan, PlanContext, Step, Constant

DEFAULT_EXAMPLE_PLANS: list[Plan] = [
    Plan(
        plan_context=PlanContext(
            query="Send hello@portialabs.ai an email with a summary of the latest news on AI",
            tool_ids=[
                "search_tool",
                "portia::google_gmail::send_email_tool",
                "portia::provider::other_tool",
            ],
        ),
        steps=[
            Step(
                task="Find and summarize the latest news on artificial intelligence",
                tool_id="search_tool",
                output="$ai_search_results",
            ),
            Step(
                task="Email $email politely with $ai_search_results",
                references=[
                    OutputReference(
                        output_id="$ai_search_results",
                        description="summary of AI news",
                    ),
                ],
                constants=[
                    Constant(
                        value="hello@portialabs.ai",
                        description="The email address to send the email to",
                    ),
                ],
                tool_id="portia::google_gmail::send_email_tool",
                output="$final_output",
            ),
        ],
    ),
    Plan(
        plan_context=PlanContext(
            query="Compare the weather of a city in the Southern hemisphere with that of a city in the Northern hemisphere. Email the results to hello@portialabs.ai.",  # noqa: E501
            tool_ids=[
                "search_tool",
                "portia::google_gmail::send_email_tool",
                "portia::provider::other_tool",
                "weather_tool",
            ],
        ),
        steps=[
            Step(
                task="What is a city in the Southern hemisphere?",
                tool_id="search_tool",
                output="$southern_hemisphere_city",
            ),
            Step(
                task="What is a city in the Northern hemisphere?",
                tool_id="search_tool",
                output="$northern_hemisphere_city",
            ),
            Step(
                task="What is the weather in the city in the input?",
                references=[
                    OutputReference(
                        output_id="$southern_hemisphere_city",
                        description="City in the southern hemisphere",
                    ),
                ],
                tool_id="weather_tool",
                output="$southern_hemisphere_city_weather",
            ),
            Step(
                task="What is the weather in the city in the input?",
                references=[
                    OutputReference(
                        output_id="$northern_hemisphere_city",
                        description="City in the norther hemisphere",
                    ),
                ],
                tool_id="weather_tool",
                output="$northern_hemisphere_city_weather",
            ),
            Step(
                task="Compare the weather of the 2 cities ($southern_hemisphere_city_weather and $northern_hemisphere_city_weather) and write a comparison summarizing the similarities and differences",  # noqa: E501
                references=[
                    OutputReference(
                        output_id="$southern_hemisphere_city_weather",
                        description="Weather of a city in the southern hemisphere",
                    ),
                    OutputReference(
                        output_id="$northern_hemisphere_city_weather",
                        description="Weather of a city in the northern hemisphere",
                    ),
                ],
                output="$weather_comparison",
            ),
            Step(
                task="Email hello@portialabs.ai with a $weather_comparison",
                references=[
                    OutputReference(
                        output_id="$weather_comparison",
                        description="Comparison of the weather in the two cities",
                    ),
                ],
                constants=[
                    Constant(
                        value="hello@portialabs.ai",
                        description="The email address to send the email to",
                    ),
                ],
                tool_id="portia::google_gmail::send_email_tool",
                output="$email_sent",
            ),
        ],
    ),
    Plan(
        plan_context=PlanContext(
            query="Send an email to hello@portialabs.ai with the weather in London",
            tool_ids=[
                "weather_tool",
                "portia::google_gmail::send_email_tool",
                "portia::provider::other_tool",
            ],
        ),
        steps=[
            Step(
                task="What is the weather in London?",
                tool_id="weather_tool",
                constants=[
                    Constant(
                        value="London",
                        description="The city from the user's query",
                    ),
                ],
                output="$london_weather",
            ),
            Step(
                task="Email $email_address politely with $london_weather",
                references=[
                    OutputReference(
                        output_id="$london_weather",
                        description="Weather in London",
                    ),
                ],
                constants=[
                    Constant(
                        value="hello@portialabs.ai",
                        description="The email address from the user's query",
                    ),
                ],
                tool_id="portia::google_gmail::send_email_tool",
                output="$email_sent",
            ),
        ],
    ),
    Plan(
        plan_context=PlanContext(
            query="If the weather in London hotter than 10C, sum it with the weather in Cairo and "
            "send the result to hello@portialabs.ai",
            tool_ids=[
                "weather_tool",
                "portia::google_gmail::send_email_tool",
                "portia::provider::other_tool",
            ],
        ),
        steps=[
            Step(
                task="Get the weather for London",
                tool_id="weather_tool",
                constants=[
                    Constant(
                        value="London",
                        description="The first city from the user's query",
                    ),
                ],
                output="$london_weather",
            ),
            Step(
                task="Get the weather for Cairo",
                tool_id="weather_tool",
                constants=[
                    Constant(
                        value="Cairo",
                        description="The second city from the user's query",
                    ),
                ],
                output="$cairo_weather",
                condition="if $london_weather is hotter than 10C",
            ),
            Step(
                task="Sum the weather in London and Cairo",
                references=[
                    OutputReference(
                        output_id="$london_weather",
                        description="Weather in London",
                    ),
                    OutputReference(
                        output_id="$cairo_weather",
                        description="Weather in Cairo",
                    ),
                ],
                output="$weather_sum",
                condition="if $london_weather is hotter than 10C",
            ),
            Step(
                task="Email $email_address politely with $weather_sum",
                references=[
                    OutputReference(
                        output_id="$weather_sum",
                        description="Sum of the weather in London and Cairo",
                    ),
                ],
                constants=[
                    Constant(
                        value="hello@portialabs.ai",
                        description="The email address from the user's query",
                    ),
                ],
                tool_id="portia::google_gmail::send_email_tool",
                output="$email_sent",
                condition="if $london_weather is hotter than 10C",
            ),
        ],
    ),
    Plan(
        plan_context=PlanContext(
            query="Get the latest messages on the Dev channel and send a summary to nathan",
            tool_ids=[
                "portia::slack::bot::list_conversation_ids",
                "portia::slack::bot::conversation_history",
                "portia::slack::bot::list_user_ids",
                "portia::slack::bot::send_message",
            ],
        ),
        steps=[
            Step(
                task="Get the id of the Dev channel",
                tool_id="portia::slack::bot::list_conversation_ids",
                output="$conversation_ids",
            ),
            Step(
                task="Get the latest messages on the Dev channel",
                references=[
                    OutputReference(
                        output_id="$conversation_ids",
                        description="The id of the Dev channel",
                    ),
                ],
                tool_id="portia::slack::bot::conversation_history",
                output="$conversation_history",
            ),
            Step(
                task="get the user id of nathan",
                tool_id="portia::slack::bot::list_user_ids",
                constants=[
                    Constant(
                        value="nathan",
                        description="The name of the user to get the id of",
                    ),
                ],
                output="$nathan_user_id",
            ),
            Step(
                task="send a summary of the conversation to nathan",
                references=[
                    OutputReference(
                        output_id="$conversation_history",
                        description="The conversation history",
                    ),
                    OutputReference(
                        output_id="$nathan_user_id",
                        description="The user id of nathan",
                    ),
                ],
                tool_id="portia::slack::bot::send_message",
                output="$message_sent",
            ),
        ],
    ),
]

"""Default examples that are passed to the query planning_agent if none are provided."""

from portia.plan import Plan, PlanContext, Step, Variable

DEFAULT_EXAMPLE_PLANS: list[Plan] = [
    Plan(
        plan_context=PlanContext(
            query="Get the temperatures in the two provided cities and then add the two temperatures together",  # noqa: E501
            tool_ids=["weather_tool", "llm_tool"],
        ),
        steps=[],
    ),
    Plan(
        plan_context=PlanContext(
            query="Compare the weather of a city in the Southern hemisphere with that of a city in the Northern hemisphere. Email the results to hello@portialabs.ai.",  # noqa: E501
            tool_ids=[
                "search_tool",
                "portia:google:gmail:send_email",
                "portia:provider::other_tool",
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
                inputs=[
                    Variable(
                        name="$southern_hemisphere_city",
                        description="City in the southern hemisphere",
                    ),
                ],
                tool_id="weather_tool",
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
                tool_id="weather_tool",
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
                tool_id="portia:google:gmail:send_email",
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
                "portia:google:gmail:send_email",
                "portia:provider::other_tool",
            ],
        ),
        steps=[
            Step(
                task="Get the weather for London",
                tool_id="weather_tool",
                output="$london_weather",
            ),
            Step(
                task="Get the weather for Cairo",
                tool_id="weather_tool",
                output="$cairo_weather",
                condition="if $london_weather is hotter than 10C",
            ),
            Step(
                task="Sum the weather in London and Cairo",
                inputs=[
                    Variable(
                        name="$london_weather",
                        description="Weather in London",
                    ),
                    Variable(
                        name="$cairo_weather",
                        description="Weather in Cairo",
                    ),
                ],
                output="$weather_sum",
                condition="if $london_weather is hotter than 10C",
            ),
            Step(
                task="Email hello@portialabs.ai with $weather_sum",
                inputs=[
                    Variable(
                        name="$weather_sum",
                        description="Sum of the weather in London and Cairo",
                    ),
                ],
                tool_id="portia:google:gmail:send_email",
                output="$email_sent",
                condition="if $london_weather is hotter than 10C",
            ),
        ],
    ),
    Plan(
        plan_context=PlanContext(
            query="Get my (john@jo.co) availability from Google Calendar tomorrow between \
              10:00 and 17:00\n- Schedule a 30 minute meeting with hello@jo.co at a time \
              that works for me",
            tool_ids=[
                "portia:google:gcalendar:check_availability",
                "portia:google:gcalendar:create_event",
            ],
        ),
        steps=[
            Step(
                task="Get the availability of john@jo.co from Google Calendar tomorrow \
                    between 10:00 and 17:00",
                tool_id="portia:google:gcalendar:check_availability",
                output="$availability",
            ),
            Step(
                task="Schedule a 30 minute meeting with hello@jo.co at a time that works for me",
                tool_id="portia:google:gcalendar:create_event",
                inputs=[
                    Variable(
                        name="$availability",
                        description="Availability of john@jo.co",
                    ),
                ],
                output="$event_created",
            ),
        ],
    ),
    Plan(
        plan_context=PlanContext(
            query="Get the latest messages on the Dev channel and send a summary to nathan",
            tool_ids=[
                "portia:slack:bot:list_conversation_ids",
                "portia:slack:bot:conversation_history",
                "portia:slack:bot:list_user_ids",
                "portia:slack:bot:send_message",
            ],
        ),
        steps=[
            Step(
                task="Get the id of the Dev channel",
                tool_id="portia:slack:bot:list_conversation_ids",
                output="$conversation_ids",
            ),
            Step(
                task="Get the latest messages on the Dev channel",
                inputs=[
                    Variable(
                        name="$conversation_ids",
                        description="The id of the Dev channel",
                    ),
                ],
                tool_id="portia:slack:bot:conversation_history",
                output="$conversation_history",
            ),
            Step(
                task="get the user id of nathan",
                tool_id="portia:slack:bot:list_user_ids",
                output="$nathan_user_id",
            ),
            Step(
                task="send a summary of the conversation to nathan",
                inputs=[
                    Variable(
                        name="$conversation_history",
                        description="The conversation history",
                    ),
                    Variable(
                        name="$nathan_user_id",
                        description="The user id of nathan",
                    ),
                ],
                tool_id="portia:slack:bot:send_message",
                output="$message_sent",
            ),
        ],
    ),
]

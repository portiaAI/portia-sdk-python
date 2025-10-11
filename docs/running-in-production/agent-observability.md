* [Running in production](/running-in-production)
* Agent observability

On this page

# Agent observability

## Using Langsmith[​](#using-langsmith "Direct link to Using Langsmith")

With Portia, you can easily instrument your agents with Langsmith in order to gain observability into the calls to the underlying language models. To do this, simply run your agent with the following environment varibles set:

```
LANGCHAIN_TRACING_V2=true  
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"  
LANGCHAIN_PROJECT=<INSERT LANGSMITH PROJECT NAME HERE>  
LANGCHAIN_API_KEY=<INSERT LANGSMITH API KEY HERE>
```

If you don't already have one, you can set up a Langsmith account by following [these instructions ↗](https://docs.smith.langchain.com/administration/how_to_guides/organization_management/create_account_api_key). This will provide you with the required project name and API key.

Once these environment variables are set, all calls that your agent makes to the underlying language models will automatically be traced within Langsmith.

## Other providers[​](#other-providers "Direct link to Other providers")

If you are using an alternative language model observability provider, please get in touch with us at [hello@portialabs.ai](mailto:hello@portialabs.ai) and let us know more about your use-case.

Last updated on **Sep 9, 2025** by **github-actions[bot]**
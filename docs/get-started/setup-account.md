* [Get started](/)
* Set up your Portia account

# Set up your Portia account

Set up your Portia cloud account. This will allow you to:

* Store and retrieve plan runs in the Portia cloud.
* Access our library of cloud hosted tools.
* Use the Portia dashboard to:
  + View your plan run history, unhandled clarifications, tool call logs.
  + Manage users, orgs and Portia API keys.

You first need to obtain a Portia API key. Head over to ([**app.portialabs.ai ↗**](https://app.portialabs.ai)) and navigate to the `Manage API keys` tab from the left hand nav. There you can generate a new API key.

On org users

You will notice a `Manage orgs and users` tab. You can set up multiple orgs in Portia. Users under the same org can all see each others' plan runs and tool call logs.

By default, Portia will look for the API key in the `PORTIA_API_KEY` environment variable. You can choose to override it for a specific `Portia` instance instance by configuring the `portia_api_key` variable as well. For now let's simply set the environment variable with the key value you generated and proceed to the next section. You can use the command below but it's always preferable to set your API keys in a .env file ultimately.

```
export PORTIA_API_KEY='your-api-key-here'
```

Upgrading your account

You can upgrade your account to a Pro plan to increase your Portia tool and plan run usage limits. Head over to the [**Billing page ↗**](https://app.portialabs.ai/dashboard/billing) to upgrade or manage your current plan.

Last updated on **Sep 9, 2025** by **github-actions[bot]**
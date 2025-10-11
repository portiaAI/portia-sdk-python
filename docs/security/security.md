* Security

On this page

# Portia cloud security

Understanding the security of your agentic system is critical to deploying agents in production. Portia uses production-grade security and encryption and offers granular customization of security policies to ensure your data is secure within our systems.

# OAuth token handling

All OAuth tokens provided to Portia from third parties are securely encrypted within the application using a unique Google KMS key per organization. No internal staff member can view decrypted token data. Once a token is expired or consumed (depending on the retention policy - see below) it is deleted from all Portia systems.

# OAuth token retention

Additionally, you can control how long Portia retains these OAuth tokens for. Within the org settings tab in the dashboard you can update your organization retention policy to one of the below options.

![Token Retention Policies](/assets/images/token_retention-894cdb11bdea15d6bd598dc182c6ecb3.png)

### Default[​](#default "Direct link to Default")

Under the default policy, Portia will store the encrypted token until it expires. We will also store any refresh tokens provided by the third party if they support this.

This retention policy is best for usability as it minimizes the number of times users will need to go through the OAuth authentication flow.

### No refresh[​](#no-refresh "Direct link to No refresh")

Under the No Refresh policy, Portia will store the encrypted tokens until they expire. However no refresh tokens will be stored.

This policy offers a blend of security and usability, with tokens being stored for a far shorter length of time (usually 24 hours though it depends on the third party).

Last updated on **Sep 9, 2025** by **github-actions[bot]**
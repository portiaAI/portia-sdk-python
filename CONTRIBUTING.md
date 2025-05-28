# Contributing to Portia SDK 🏗️

Thank you for your interest in contributing to Portia SDK! We welcome contributions that improve the library and help us build a great experience for the community.

## What to contribute
* **Paid issue contributions** We post paid contributions to our issues list for which we will remunerate contributions. Please see the guidelines [below]((https://github.com/portiaAI/portia-sdk-python/CONTRIBUTING.md#paid-contributions).
* **Documentation** Tutorials, how-to guides and revisions to our existing docs go a long way in making our repo easier to setup and use
* **Examples** Show our community what you can do with our SDK. We particularly encourage end-to-end, real-world applications
* **Bug reports** Please include a detailed method to reproduce any bugs you spot. We would be grateful if you give the [Issue Tracker](https://github.com/portiaAI/portia-sdk-python/issues) a quick skim first to avoid duplicates 🙌
* **Bug fixes** Those are our favourites! Please avoid breaking changes. The next section has some helpful tips for that.
* **Feedback** Help us be better. Come chat to us on [Discord](https://discord.gg/DvAJz9ffaR) about your experience using the SDK 🫶

⚠️ **A note on new features** If you have something in mind, please give us a shout on our Discord channel. Features like new core abstractions, changes to infra or to dependencies will require careful consideration before we can move forward with them.

## Paid contributions
Paid contributions are shown in the [Issue](https://github.com/portiaAI/portia-sdk-python/issues) list with the monetary amount shown as follows [£20]. If an issue does not include a monetary amount, it indicates that fixing it will not be remunerated (though we appreciate it greatly!). Please bear in mind the following rules for paid contributions:
* If you wish to work on a paid contribution you should assign it to yourself. You then have a week to submit the code for the issue or else it will be unassigned from you. If the code is not submitted in this time, and you are unassigned, you are not permitted to assign yourself to that issue again.
* Contributors can have only a single assigned issue at a time.
* We expect paid contributions to require no more than 1 major review (i.e broader suggestions on direction), and 4 minor reviews. If more than this is required, the contribution will not be remunerated.
* Once your feature is ready for review, please assign emmaportia as a reviewer.

### Getting paid
* Once you have contributed and submitted a paid contribution, please email code-submission@portialabs.ai including a link to the PR you made and a screenshot of the Github accounts profile page that authored the PR to prove your identiy. If you are using a different currency we will provide the remuneration at the local exchange rate.

## How to contribute

1. **Fork the Repository**: Start by forking the repository and cloning it locally.
2. **Create a Branch**: Create a branch for your feature or bug fix. Use a descriptive name for your branch (e.g., `fix-typo`, `add-feature-x`).
3. **Install the dependencies** We use uv to manage dependencies. Run ``uv install --all-extras``
4. **Make Your Changes**: Implement your changes in small, focused commits. Be sure to follow our linting rules and style guide.
5. **Run Tests**: If your changes affect functionality, please test thoroughly 🌡️ Details on how run tests are in the **Tests** section below.
6. **Lint Your Code**: We use [ruff](https://github.com/charliermarsh/ruff) for linting. Please ensure your code passes all linting checks. We prefer per-line disables for rules rather than global ignores, and please leave comments explaining why you disable any rules.
7. **Open a Pull Request**: Once you're happy with your changes, open a pull request. Ensure that your PR description clearly explains the changes and the problem it addresses. The **Release** section below has some useful tips on this process.
8. **Code Review**: Your PR will be reviewed by the maintainers. They may suggest improvements or request changes. We will do our best to review your PRs promptly but we're still a tiny team with limited resource. Please bear with us 🙏
10. **Merge Your PR**: Once approved, the author of the PR can merge the changes. 🚀

## Linting

We lint our code using [Ruff](https://github.com/astral-sh/ruff). We also have [pre-commit](https://pre-commit.com/) setup to allow running this easily locally.

## Tests

We write two types of tests:
- Unit tests should mock out the LLM providers, and aim to give quick feedback. They should mock out LLM providers.
- Integration tests actually call LLM providers, are much slower but test the system works fully.

To run tests:
- Run all tests with `uv run pytest`.
- Run unit tests with `uv run pytest tests/unit`.
- Run integration tests with `uv run pytest tests/integration`.

We utilize [pytest-parallel](https://pypi.org/project/pytest-parallel/) to execute tests in parallel. You can add the `--workers=4` argument to the commands above to run in parallel. If you run into issues running this try setting `export NO_PROXY=true` first.

## Release

Releases are controlled via Github Actions and the version field of the `pyproject.toml`. To release:

1. Create a PR that updates the version field in the `pyproject.toml`.
2. Merge the PR to main.
3. Github Actions will create a new tag and push the new version to PyPi.

## Contributor License Agreement (CLA)

By submitting a pull request, you agree to sign our Contributor License Agreement (CLA), which ensures that contributions can be included in the project under the terms of our current [license](https://github.com/portiaAI/portia-sdk-python/edit/main/CONTRIBUTING.md#:~:text=CONTRIBUTING.md-,LICENSE,-Logo_Portia_Stacked_Black.png). We will ask you to sign this CLA when submitting your first contribution.

## Thank you

Thank you for contributing to Portia SDK Python!

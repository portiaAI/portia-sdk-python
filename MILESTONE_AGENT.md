# Milestone Agent

## What is it?

This is an experiment that came out of getting frustrated with the limited flexibility of Portia's current rigid execution paradigm.

The goal is to blend Portia's USP of adding structure to agentic workflows, but bringing in the flexibility of a ReAct (reason -> act) agentic loop.

The idea is to define high level structure as "milestones" using a PlanBuilder approach but then have a reason-acting (ReAct) agentic loop execute each milestone.

![Milestone Agent](milestone-agent.png)

## Where is it?

The milestone plan and builder classes is in the `milestone_plan.py` module.

The agent implementation is in the `milestone_agent.py` module (this is a copy/paste/modify of the `OneShotAgent`).

A couple of use-cases are in the `milestone_agent_examples.py` module.

## Why?

### Why ReAct?

There is a lot of power in reason-act (ReAct) in terms of flexibility and error handling. It gives responsibility to the agent to solve certain problems that are hard or intractable in our current fixed execution paradigm:
- Replanning: the agent is constantly replanning
- Unrolling: the agent knows to keep iterating if there is more data, more pages, more items to process.
- Memory: the agent can be given tools to work with large objects and use them when approprioate (which is not knowable at the planning state in many cases)


### Why milestones?

This approach is not as restrictive as current planning paradigm, but more controlled than pure agentic loop. Specifically, it still allows users to deterministically define that certain things will happen before others.

The hypothesis is that the domain experts who understand the business processes that will be automated by agents understand their processes in terms of high level steps (milestones). And this works both ways - the milestone progress makes it easier for users to reason about the agents state and progress. 

As an example of how this makes it easier to reason about the agent when building it: it makes it easier to think about the offramps. What if the customer email is not a refund request email? What if the customers order cannot be found? If these steps are implemented as milestones, the plan can encode what happens if the milestone is not achieved.

The implementation of each step can be a mix and match of different exeuction paradigms - some milestones might be executed as Agents, others as custom user code, others as Agentic workflows (i.e. current Portia). The user can choose the best approach for each step. They can also choose the granularity of the milestones - some milestones might be very granular and only have access to 1-2 tools, others might be more complex and have access to many tools.

The execution can be constrained in similar ways that we define now:
- Clarification before certain tools are used
- Execution hooks

But also:
- restrictions on the number of times a tool can be called
- restrictions on the maximum number of iterations within a milestone
- restrictions on the maximum number of input/output tokens
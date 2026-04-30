The next feature I would like to work on is the parquet file creation so that the output data can be uploaded to huggingface in the way that is most useful for machine learning experiments. 

To create the spec for the feature interview me in depth about every aspect of this plan until we reach a shared understanding. Walk down each branch of the design tree, resolving dependencies between decisions one by one.  Ask about requirements, edge cases, user experience, data models, and failure modes. Do not write a plan document or code until we are in agreement about how to proceed. 

After the interview phase, and before you start work on this project, create three files: 
1. spec.md — a complete spec with goals, implementation details, and a verification section describing exactly how you'll prove each piece works.
2. todo.md — a running to-do list you'll edit as you work. Break complex tasks into verifiable sub-tasks.
3. Store tests in tests/ to verify everything you build. Loop on them until each passes.

While you work:
 (a) Consult spec.md before every change.
 (b) Mark each completed task in todo.md with [x] once it is completed. 
 (c) run tests after every meaningful commit, 
 (d) every 20 iterations or so, call a fresh sub-agent with "Review spec.md and the current implementation for gaps" and loop on the sub-agent's feedback until alignment is reached.

Do not ask me for clarification on anything you can resolve by reading the spec and running the tests. Start with the spec.

# Process to follow

## Outline of the feature
The next feature is a script called huggingface.py.
At a minimum it will provide commands to upload, download and tag specific datasets in huggingface. 
Other capabilities may be discovered in the interview stage - please suggest those that may be useful. 
A dataloading script - which will probably be separate can also be discussed. This would be a script that is useful to users of the repo.
It would help in creating subsets of the data such as for training, testing and validation of models. 

## Interview stage
To create the spec for the features interview me in depth about every aspect of this plan until we reach a shared understanding. Walk down each branch of the design tree, resolving dependencies between decisions one by one.  Ask about requirements, edge cases, user experience, data models, and failure modes. Do not write a plan document or code until we are in agreement about how to proceed. 

## File creation
After the interview phase, and before you start work on this project, create these two files in the `huggingface_feature` folder (the folder that contains this file):
1. `spec.md` — a complete spec with goals, implementation details, and a verification section describing exactly how you'll prove each piece works.
2. `todo.md` — a running to-do list you'll edit as you work. Break complex tasks into verifiable sub-tasks.
3. Store tests in tests/ to verify everything you build. Loop on them until each passes.

## Long running phase once the plan is ready - follow these steps while you work:
 (a) Consult spec.md before every change.
 (b) Mark each completed task in todo.md with [x] once it is completed. 
 (c) Run tests after every meaningful commit, 
 (d) Every 20 iterations or so, call a fresh sub-agent with "Review spec.md and the current implementation for gaps" and loop on the sub-agent's feedback until alignment is reached.

Do not ask me for clarification on anything you can resolve by reading the spec and running the tests. Start with the spec.

### Here's an overview of how HuggingFace handles dataset versioning:

Short answer: HuggingFace does not have a formal "version" concept built into the Hub UI the way PyPI or npm do. Versioning is done through Git tags, which are lightweight and well-supported.

### How versioning works

Every HuggingFace dataset repo is a Git repository. The two main mechanisms are:

1. Git tags (recommended for versioning)

After pushing your updated dataset, create a tag to mark the version:

from huggingface_hub import create_tag

create_tag(
"DavidCBaines/ebible_corpus",
tag="v2.0",           # or "2026-05", a date, whatever convention you prefer
tag_message="Updated corpus with ...",
repo_type="dataset"
)

Users can then load a specific version:
from datasets import load_dataset
ds = load_dataset("DavidCBaines/ebible_corpus", revision="v2.0")

2. Git commits (automatic)

Every push creates a commit in the repo's history, visible under the "Files and versions" → "History" tab. This gives you full revision history even without explicit tags.

---
Recommended update process

1. Update your files — push new Parquet files and updated README.md (dataset card) using push_to_hub() or huggingface_hub.upload_file() / upload_folder().
2. Tag the previous version first (optional but good practice) — if users rely on the current data, tag v1.0 before you overwrite it, so they can still pin to it.
3. Tag the new version — after pushing, call create_tag("DavidCBaines/ebible_corpus", tag="v2.0", repo_type="dataset").
4. Update the dataset card — note the version history in your README.md under a ## Changelog or ## Versions section. This is the most visible place for users.

---
What NOT to do

- There is no separate "Versions" tab or release concept like GitHub Releases for datasets.
- Tags on main are the more common practice and simpler to manage.

---
### Practical suggestion for this case

Since you're uploading a new version of ebible_corpus, I'd suggest:

1. Tag the current main as v1.0 now (to preserve it)
2. Push your new Parquet + updated dataset card to main
3. Tag main as v2.0
4. Add a short changelog to the dataset card README

Would you like help writing the upload script using huggingface_hub or datasets.push_to_hub()? I can also look at your existing corpus_to_parquet.py to see how much of this is already in place.

Sources:
- https://huggingface.co/docs/hub/datasets-adding
- https://github.com/huggingface/datasets/discussions/5370
- https://discuss.huggingface.co/t/how-exactly-does-datasets-versioning-work/20853
- https://discuss.huggingface.co/t/tagging-with-git-tags-models-for-versioning/78489


Using Python: 
from huggingface_hub import HfApi

api = HfApi()  # reads token from hf auth login, or pass token= explicitly

# Step 1: Tag the CURRENT state as v1.0 (do this BEFORE uploading)
api.create_tag(
"DavidCBaines/ebible_corpus",
tag="v1.0",
tag_message="Original corpus release",
repo_type="dataset"
)

# Step 2: Upload new Parquet + dataset card
api.upload_folder(
folder_path="C:/ebible_data/huggingface",   # local folder with your files
repo_id="DavidCBaines/ebible_corpus",
repo_type="dataset",
commit_message="Update corpus v2.0"
)

# Step 3: Tag the new state as v2.0
api.create_tag(
"DavidCBaines/ebible_corpus",
tag="v2.0",
tag_message="Updated corpus - <describe what changed>",
repo_type="dataset"
)

---

Sources:
- https://huggingface.co/docs/hub/datasets-adding
- https://huggingface.co/docs/huggingface_hub/guides/upload
- https://github.com/huggingface/datasets/discussions/5370
- https://huggingface.co/docs/huggingface_hub/guides/cli


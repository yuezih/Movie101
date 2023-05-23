# CNCLIP+RNL

**Since the pipeline is complex, it's a lot of work to organize the code and I'm still working on that. If you urgently need it, please let me know.**

The goal of the Temporal Narration Grounding (TNG) task is to locate the start and end times of a relevant segment in an entire movie given a piece of narration text. The baseline provided in this paper divides this process into two stages. The first is a coarse-grained Global Shot Retrieval, which initially divides the movie into equal-length segments of 20 seconds each and then retrieves the most relevant segment. The second is a fine-grained Local Temporal Grounding, which involves precise localization around the retrieved segment. Specifically, the retrieved 20-second segment is expanded to 200 seconds, and then temporal localization is performed.

# Global Shot Retrieval

This stage is achieved by CNCLIP, a Chinese image-text pre-training model, which we first need to make video retrieval-capable. Therefore, we constructed a temporary dataset Movie101-R (temp) for training a retrieval model, as described in the paper. Then, CNCLIP is fine-tuned on Movie101-R (temp).

# Local Temporal Grounding

This stage is achieved by RNL. 
# Lung Cancer Subtyping and Triage

## The problem

I am building the triage model for a digital pathology lab. Given one
hematoxylin and eosin tissue patch from a lung whole slide image, I have to do
two things at once: output a belief distribution over seven diagnostic classes,
and output a referral score saying how badly this patch needs a senior
pathologist to look at it.

The lab can only escalate a fixed fraction of cases. An escalated patch gets
resolved by the specialist and takes a fixed score but burns scarce expert time.
An auto-reported patch is scored on diagnostic harm, and a confident mistake on
the wrong side of the cancer boundary is punished heavily. The seven classes form
a clinical hierarchy and the scoring respects that hierarchy rather than treating
them as flat and interchangeable.

## What I did

The points here do not come from a bigger network. There are only about 472
images across 30 patients, so training from scratch or fine-tuning a large model
memorizes patients and then collapses on the disjoint test patients. Instead I
use frozen, publicly downloadable pathology foundation model embeddings with a
small heavily regularized head on top, validated with patient-grouped CV.

The other half of the score is the decision layer. The metric is fully specified,
so rather than picking a threshold by feel I optimize the referral rule directly
against the exact scoring function, including the escalation budget and the
asymmetric penalty around the cancer boundary.

## Layout

`solution.py` is the entry point, `dev/` holds the training and probe scripts,
`Approach.md` is the strategy write up and `notes.md` the running log. Datasets
are not committed.

"""
Weights & Biases (W&B) Quick Reference
======================================

This module stores a long documentation string summarizing the core features of
the `wandb` Python SDK.  It is intended to provide a convenient source of
context for code assistants that do not have network access.  The content in
this string is derived from the official W&B documentation as of July 24 2025
and is meant for human and machine consumption.  If you paste this file into
your project the assistant can refer to it when generating code without
re-downloading the documentation.  The summary covers setup, runs, logging,
environment variables, artifacts, and hyperparameter sweeps.

Overview
--------

Weights & Biases is an AI developer platform for tracking, visualizing and
managing machine‑learning experiments.  The `wandb` Python package lets you
instrument training scripts and notebooks to log metrics, store hyperparameters
and share results.  A **run** is the basic unit of computation: each run
represents a single execution of your training script.  Runs are grouped into
projects (and optionally entities, which correspond to your user or team).  You
initialize a run with ``wandb.init()`` and log metrics with ``wandb.log`` or
``run.log``.  At the end of training call ``run.finish()`` to cleanly mark the
run as finished【276339629799248†L482-L491】.

Setup and authentication
------------------------

1. **Sign up and API key** – Create an account at `wandb.ai` and generate an
   API key from your profile.  You can authenticate either by setting the
   ``WANDB_API_KEY`` environment variable or by calling ``wandb.login()``.  If
   you set ``WANDB_API_KEY=$YOUR_API_KEY`` before running your script, the
   library will use that key to authenticate【814481924530985†L268-L299】.

2. **Installation** – Install the client with ``pip install wandb``.  To log in
   interactively call ``wandb.login()`` in Python or run ``wandb login`` from
   the command line【356384173637991†L329-L345】.

3. **Environment variables** – W&B respects many environment variables.  Some
   commonly used ones are:

   - ``WANDB_API_KEY`` – your authentication key; required if you haven’t run
     ``wandb login``【814481924530985†L309-L313】.
   - ``WANDB_PROJECT`` – default project name used when initializing runs【814481924530985†L344-L347】.
   - ``WANDB_ENTITY`` – user or team name to associate with runs【814481924530985†L344-L347】.
   - ``WANDB_NAME`` – human‑readable run name; overrides randomly generated names
    【814481924530985†L360-L361】.
   - ``WANDB_NOTES`` – multi‑line notes or Markdown description for the run【814481924530985†L360-L365】.
   - ``WANDB_MODE`` – set to ``'online'`` (default), ``'offline'`` (log locally
     without syncing) or ``'disabled'`` to turn off the client entirely【814481924530985†L357-L359】.
   - ``WANDB_DIR`` and ``WANDB_ARTIFACT_DIR`` – directories where runs and
     downloaded artifacts are stored【814481924530985†L326-L333】.
   - ``WANDB_CONSOLE`` – set to ``'off'`` to disable redirecting stdout/stderr to
     the W&B interface【814481924530985†L320-L321】.
   - ``WANDB_RUN_ID`` – custom unique ID for a run; useful for resuming
     experiments【814481924530985†L374-L377】.
   - ``WANDB_RESUME`` – control automatic resume behavior; values: ``'never'``
     (default), ``'allow'``, ``'must'``【814481924530985†L368-L370】.
   - ``WANDB_DISABLE_GIT`` and ``WANDB_DISABLE_CODE`` – disable Git or code
     saving if you don’t want W&B to capture your repository or notebook
    【814481924530985†L336-L339】.

Initializing and managing runs
------------------------------

### Creating a run

Call ``wandb.init`` at the start of your training script to create a run.  The
most important arguments are:

* ``project`` – name of the project.  Runs are grouped by project.  You can
  override the default from ``WANDB_PROJECT``.
* ``entity`` – user or team under which the run lives.  This corresponds to
  your W&B username or organization.  A default can be set via
  ``WANDB_ENTITY``.
* ``config`` – a dictionary of hyperparameters and metadata.  The contents of
  ``config`` appear in the W&B UI and can be used later for sweeps.  Example:

    ```python
    run = wandb.init(project="my-awesome-project", config={
        "learning_rate": 0.01,
        "epochs": 10,
    })
    ```
    【356384173637991†L329-L340】

``wandb.init`` returns a ``Run`` object.  You can access the configuration via
``run.config`` (or the global ``wandb.config``).  Each call to ``wandb.init``
creates a new run; call it once per training script.  If you want to
manually specify a run ID, pass ``id=<string>`` or set ``WANDB_RUN_ID``【276339629799248†L540-L547】.

### Logging metrics

During training, use ``run.log`` or the alias ``wandb.log`` to record metrics.
Pass a dictionary mapping metric names to values.  Logging inside a loop
creates a time series; include a step indicator (e.g., ``epoch`` or ``batch``)
if you want to control the x‑axis in the UI:

```python
for epoch in range(epochs):
    # compute metrics
    run.log({"accuracy": acc, "loss": loss, "epoch": epoch})
```【276339629799248†L359-L379】.  You can also log rich media (images,
audio, videos, tables) by constructing W&B media objects and passing them to
``log`` (see below).

### Finishing a run

Call ``run.finish()`` when training completes to mark the run as finished and
ensure that data is fully synced【276339629799248†L482-L491】.  If you forget to
call ``finish``, the run may continue in the background.

### Run names, tags and grouping

* **Name** – pass ``name=<string>`` to ``wandb.init`` or set ``WANDB_NAME`` to
  give the run a human‑readable name【276339629799248†L573-L579】.
* **Tags** – pass ``tags=[...]`` to ``wandb.init`` or set ``WANDB_TAGS`` as
  comma‑separated values to label runs.
* **Group** – pass ``group=<experiment-name>`` or set ``WANDB_RUN_GROUP`` to
  group related runs together (useful for sweeps or multi‑run jobs).

Logging media
-------------

W&B can log many data types in addition to scalar metrics.  Construct media
objects and include them in ``wandb.log``.  Here are some common examples:

* **Images** – pass a NumPy array, PIL Image or file path to ``wandb.Image``.

  ```python
  # Log a batch of images from a tensor
  images = [wandb.Image(img_array, caption="Prediction vs. input") for img_array in batch]
  run.log({"examples": images})
  ```
  When logging images, W&B infers grayscale (1 channel), RGB (3 channels) or
  RGBA (4 channels) based on the array shape and normalizes float arrays
  automatically【186279069538048†L331-L344】.  You can instead supply a PIL image or
  save your own images to disk and log them by filename【186279069538048†L349-L365】.

* **Segmentation masks** – to overlay semantic masks on an image, pass a
  ``masks`` dictionary to ``wandb.Image``.  Each key corresponds to a mask
  type (e.g., ``"predictions"`` or ``"ground_truth"``) and maps to a
  dictionary containing ``mask_data`` (a 2‑D array of integer labels) and
  optional ``class_labels`` (mapping label IDs to names)【186279069538048†L380-L407】.

* **Bounding boxes** – pass a ``boxes`` dictionary with ``box_data`` (list of
  dictionaries specifying positions, class IDs, scores, etc.) and optional
  ``class_labels``.  Box positions can be specified either by fractional
  coordinates (``minX``, ``maxX``, ``minY``, ``maxY``) or by pixel coordinates
  (``middle``, ``width``, ``height``)【186279069538048†L433-L456】.  Example:

  ```python
  class_id_to_label = {1: "car", 2: "road", 3: "building"}
  img = wandb.Image(
      image,
      boxes={
          "predictions": {
              "box_data": [
                  {"position": {"minX": 0.1, "maxX": 0.2, "minY": 0.3, "maxY": 0.4},
                   "class_id": 2,
                   "box_caption": class_id_to_label[2],
                   "scores": {"acc": 0.1, "loss": 1.2}},
                  # more boxes here
              ],
              "class_labels": class_id_to_label,
          }
      },
  )
  run.log({"scene": img})
  ```

* **Tables** – create a ``wandb.Table`` with columns and rows to log tabular
  data.  Tables can include images and other media types.
* **Other media** – W&B provides ``wandb.Audio``, ``wandb.Video``,
  ``wandb.Plotly``, and other classes for logging audio clips, videos,
  interactive plots, molecular structures, etc.  See the official docs for
  details.

Artifacts
---------

Artifacts provide dataset and model versioning.  They allow you to save files,
directories or references to external URIs and then reuse them in later runs.

1. **Creating an artifact** – call ``artifact = wandb.Artifact(name, type)``.
   The ``type`` is an arbitrary string such as ``"dataset"`` or ``"model"``.
2. **Adding files** – use ``artifact.add_file(local_path)`` to add a single
   file or ``artifact.add_dir(local_path)`` to add an entire directory【625377533395665†L398-L440】.
   You can also use ``artifact.add_reference(uri, name="optional")`` to track
   external resources like S3 objects or HTTP URLs【625377533395665†L470-L503】.
3. **Logging the artifact** – create a run and call ``run.log_artifact(artifact)``
   to upload it【625377533395665†L345-L374】.  Uploads happen asynchronously.
4. **Using artifacts** – call ``run.use_artifact(name:version)`` to download
   an artifact in later runs.

Hyperparameter sweeps
---------------------

W&B Sweeps automate hyperparameter optimization by launching many runs with
different settings.  A **sweep configuration** can be defined in YAML or as a
Python dictionary.  Key top‑level keys include:

* ``program`` – path to the training script (used when running from the CLI).
* ``method`` – search algorithm: ``random``, ``grid``, ``bayes`` (Bayesian
  optimization) or ``hyperband``.
* ``metric`` – dictionary with ``name`` (metric to optimize) and ``goal``
  (``"minimize"`` or ``"maximize"``)【125609980343997†L292-L317】.
* ``parameters`` – dictionary describing the hyperparameters to sweep over.
  Each entry can specify ``values`` (explicit list), ``min``/``max`` for a
  range, or a probability ``distribution``.  Example config:

  ```yaml
  program: train.py
  name: sweepdemo
  method: bayes
  metric:
    goal: minimize
    name: validation_loss
  parameters:
    learning_rate:
      min: 0.0001
      max: 0.1
    batch_size:
      values: [16, 32, 64]
    epochs:
      values: [5, 10, 15]
    optimizer:
      values: ["adam", "sgd"]
  ```
  【125609980343997†L292-L324】

  In Python you can define the same configuration as a dictionary:

  ```python
  sweep_config = {
      "name": "sweepdemo",
      "method": "bayes",
      "metric": {"goal": "minimize", "name": "validation_loss"},
      "parameters": {
          "learning_rate": {"min": 0.0001, "max": 0.1},
          "batch_size": {"values": [16, 32, 64]},
          "epochs": {"values": [5, 10, 15]},
          "optimizer": {"values": ["adam", "sgd"]},
      },
  }
  ```【125609980343997†L335-L347】

To launch a sweep:

1. Initialize a sweep: ``sweep_id = wandb.sweep(sweep_config, project="my-project")``.
2. Define a training function that reads hyperparameters from ``wandb.config``.
3. Start an agent to run jobs: ``wandb.agent(sweep_id, function=train, count=<n>)``.

Each agent call runs the training function with a different set of parameters.

Miscellaneous tips
------------------

* **Offline mode** – set ``WANDB_MODE=offline`` to log runs locally without
  syncing to the cloud【814481924530985†L291-L299】.  You can later call
  ``wandb.sync`` to upload offline runs.
* **Anonymous runs** – set ``WANDB_ANONYMOUS=allow`` to create runs without a
  logged‑in account【814481924530985†L306-L309】.
* **Disable console output** – set ``WANDB_QUIET`` or ``WANDB_SILENT`` to
  suppress logs printed to stdout【814481924530985†L378-L383】.
* **Resuming runs** – set ``WANDB_RESUME=allow`` and pass a custom
  ``WANDB_RUN_ID`` to resume a previous run if your script crashes【814481924530985†L368-L377】.
* **Saving code** – by default W&B captures Git state and diffs; disable by
  setting ``WANDB_DISABLE_GIT=true`` or ``WANDB_DISABLE_CODE=true``【814481924530985†L336-L339】.

This reference provides a concise overview of the W&B SDK.  For deeper
information—such as logging histograms, customizing charts, or managing data
governance—consult the official documentation.
"""

# The documentation string can be imported from this module:
wandb_docs: str = __doc__  # type: ignore

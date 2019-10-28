# Good Things To Know

Here I will document the various procedures required to do things in this repository.

### Running An Experiment

If debugging an experiment, run the following:

```bash
python ./cow_tus/run.py -F EXPERIMENT_DIR -u -p
```

If running a real experiment, run the following:

```bash
python ./cow_tus/run.py -F EXPERIMENT_DIR -p -e -c "This is a comment"
```
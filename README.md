# marbl material proporties mini-miniapp

## building

Just run `make`. On sierra nodes this will build a exe that can eval on the cpu or gpu.

## running

See `./mmp-$SYS_TYPE -h` for options.

By default the evals will be on the cpu, if you want to run on the gpu: `./mmp-$SYS_TYPE -d cuda`

## questions

- The indicators are constant but they change in a real sim with ale, should they change here?
- The initial eos inputs are random and unchanging, should they be real data and/or change
  each "cycle"?

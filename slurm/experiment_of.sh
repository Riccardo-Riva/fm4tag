#!/bin/bash
# experiment_of <path>: the campaign name of a file under experiments/<campaign>/ (printed), or EXPERIMENT if set,
# or nothing.  Sourced by the launchers so that every output lands in experiments/<campaign>/{outputs,plots}/.
experiment_of() {
    if [[ -n "${EXPERIMENT:-}" ]]; then echo "${EXPERIMENT}"; return; fi
    local p
    p=$(realpath -m "$1")
    if [[ "$p" == */experiments/*/* ]]; then
        p=${p#*/experiments/}
        echo "${p%%/*}"
    fi
}

"""Shell completion scripts. Each asks `cimo __complete WORDS...` for candidates, so completion
always matches the installed command tree; a last line `__file__` or `__dir__` asks the shell
to complete paths."""

BASH = r'''# cimo completion for bash: add to ~/.bashrc, or: source <(cimo completion bash)
_cimo() {
    local cur=${COMP_WORDS[COMP_CWORD]} IFS=$'\n' kind=
    local out=($(cimo __complete "${COMP_WORDS[@]:1:COMP_CWORD}" 2>/dev/null))
    if (( ${#out[@]} )) && [[ ${out[${#out[@]}-1]} == __*__ ]]; then
        kind=${out[${#out[@]}-1]}
        unset "out[${#out[@]}-1]"
    fi
    COMPREPLY=($(compgen -W "${out[*]}" -- "$cur"))
    case $kind in
        __file__) COMPREPLY+=($(compgen -f -- "$cur")) ;;
        __dir__) COMPREPLY+=($(compgen -d -- "$cur")) ;;
    esac
}
complete -o filenames -o bashdefault -F _cimo cimo
'''

ZSH = r'''#compdef cimo
# cimo completion for zsh: save as _cimo in a directory on $fpath, or: source <(cimo completion zsh)
_cimo() {
    local -a out
    local kind
    out=("${(@f)$(cimo __complete "${words[@]:1:$((CURRENT-1))}" 2>/dev/null)}")
    out=(${out:#})
    if [[ ${out[-1]} == __*__ ]]; then
        kind=${out[-1]}
        out=("${(@)out[1,-2]}")
    fi
    (( ${#out} )) && compadd -a out
    case $kind in
        __file__) _files ;;
        __dir__) _files -/ ;;
    esac
}
if [[ $zsh_eval_context[-1] == loadautofunc ]]; then
    _cimo "$@"
else
    compdef _cimo cimo
fi
'''

FISH = r'''# cimo completion for fish: save as ~/.config/fish/completions/cimo.fish
function __cimo_complete
    set -l words (commandline -opc)
    set -e words[1]
    set -l cur (commandline -ct)
    for line in (cimo __complete $words "$cur" 2>/dev/null)
        switch $line
            case __file__
                __fish_complete_path "$cur"
            case __dir__
                __fish_complete_directories "$cur"
            case '*'
                echo $line
        end
    end
end
complete -c cimo -f -a '(__cimo_complete)'
'''

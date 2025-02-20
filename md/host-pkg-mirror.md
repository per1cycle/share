# Preface
I was deeply impressed by how clean freebsd is designed.

# Table of content
- prepare
- poudriere
- pkg
- usage

## prepare
you need a amd64(x86)/powerpc64 as host, cause poudriere need qemu-user-static while cross build for riscv target.
reason:
https://github.com/freebsd/freebsd-ports/blob/main/emulators/qemu-user-static/Makefile#L15

```zsh
pkg update
pkg install poudriere qemu-user-static
```

btw, jail is also needed for the whole process
```
# sysrc jail_enable="YES"
# sysrc jail_parallel_start="YES"
```

## poudriere
on your x86 host

## pkg
upload to your vps with web server nginx as an example here.

## usage
the sample configuration:
```
pericycle: {
        url: "http://bsd.per1cycle.org/pkg/${ABI}/quarterly",
        mirror_type: "none",
        enabled: yes
}
```
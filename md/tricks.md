# Tricks

## Patches

### b4 work flow

how to use b4 to get involved in kernel development?
```sh
git clone <kernel repo >
#       name      fork from
b4 prep -n branch -f master
b4 prep --edit-cover # edit cover of your patch series.
# after modify the code.

b4 send -o /tmp/tosend # show your patch here.
b4 send --reflect # send mail to yourself
b4 send 
```


### How to modify the previous commit, keep the commit message but modify the code?

```sh
git rebase -i HEAD~3
# set e in front of the commit you want modify
# after do the change.
git add .
git commit --ammend -s 
git rebase --continue

```
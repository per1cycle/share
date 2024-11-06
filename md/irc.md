# get start into irc workflow

> Don't panic.

## introduct irc client weechat

[official repo](https://weechat.org/)

### install

on debian 12
```bash
sudo apt install weechat
```

### setup 

use command `weechat` to geeeeeeet in the weechat irc client.

### get directly into irc channel(not recommend)

TODO

### use bouncer(recommend)

see znc 

## introduce irc bouncer znc 
```bash
sudo apt install weechat
# use `znc --makeconf` to generate a znc config for irc
```

### settttup znc config
e.g. in this example we use irc.oftc.net as irc network, you can choose irc.libre.net in same way, only different is the identify way.

```text
[ .. ] Checking for list of available modules...
[ ** ] 
[ ** ] -- Global settings --
[ ** ] 
[ ?? ] Listen on port (1025 to 65534): 10025
[ ?? ] Listen using SSL (yes/no) [no]: yes // perfer, notice use the irc network's port.
 ?? ] Listen using both IPv4 and IPv6 (yes/no) [yes]: // use default
[ .. ] Verifying the listener...
[ ** ] Unable to locate pem file: [/home/z/.znc/znc.pem], creating it
[ .. ] Writing Pem file [/home/z/.znc/znc.pem]...
[ ** ] Enabled global modules [webadmin]
[ ** ] 
[ ** ] -- Admin user settings --
[ ** ] 
[ ?? ] Username (alphanumeric): pericycle // in this case, i have irc account in oftc, so i just use my name.
[ ?? ] Enter password: 
[ ?? ] Confirm password: 
// use default
[ ?? ] Nick [pericycle]: 
[ ?? ] Alternate nick [pericycle_]: 
[ ?? ] Ident [pericycle]: 
[ ?? ] Real name (optional): 
[ ?? ] Bind host (optional): 

[ ** ] Enabled user modules [chansaver, controlpanel]
[ ** ] 
[ ?? ] Set up a network? (yes/no) [yes]:   
[ ** ] 
[ ** ] -- Network settings --
[ ** ] 
[ ?? ] Name [freenode]: 

// https://www.oftc.net/
[ ?? ] Name [freenode]: irc.oftc.net
[ ?? ] Name [freenode]: oftc
[ ?? ] Server host (host only): irc.oftc.net
[ ?? ] Server uses SSL? (yes/no) [no]: no // buggy with ssl.... on my computer.
[ ?? ] Server port (1 to 65535) [6667]: 
[ ?? ] Server password (probably empty): 
[ ?? ] Initial channels: #alpine-devel
[ ** ] Enabled network modules [simple_away]

```

now the bouncer is setup already.

how to use it?

```bash
weechat

# now you are in weechat cli
/server add oftc localhost/10025
/set irc.server.oftc.username pericycle
/set irc.server.oftc.password <my password, don't look, u dum as>
/save
# do this step after you register your nickname.
# /set irc.server.oftc.command "/msg nickserv identify xxx"

```

reference:
- https://weechat.org/files/doc/weechat/stable/weechat_quickstart.en.html

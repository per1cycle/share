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
> difference between oftc and libera, oftc has channels like debian*, alpine* and libera has channel like u-boot, gentoo*, fedora*, etc.

```bash
# this time we use libera.chat server as an example.
# in weechat client:
/server add libera irc.libera.chat/6667
/set irc.server.libera.username FOOBARTEST
/connect libera
/msg Nickserv register <password> <email>
/join #gentoo
dlan: It works this will at dlan it work ;-)
```

### use bouncer(recommend)

see [znc](#znc) 

## znc 
```bash
sudo apt install weechat
# use `znc --makeconf` to generate a znc config for irc
```

### settttup znc config
e.g. in this example we use `irc.oftc.net` as irc network, you can choose `irc.libera.chat` in same way, only different is the identify way.

```text
[ .. ] Checking for list of available modules...
[ ** ] 
[ ** ] -- Global settings --
[ ** ] 
[ ?? ] Listen on port (1025 to 65534): 10025
[ ?? ] Listen using SSL (yes/no) [no]: no // perfer is yes, but buggy on my device. notice use the irc network's port.
 ?? ] Listen using both IPv4 and IPv6 (yes/no) [yes]: // use default
[ .. ] Verifying the listener...
[ ** ] Unable to locate pem file: [/home/z/.znc/znc.pem], creating it

[ ** ] -- Admin user settings --
[ ** ] 
[ ?? ] Username (alphanumeric): pericycle 
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
[ ?? ] Name [freenode]: oftc
[ ?? ] Server host (host only): irc.oftc.net
[ ?? ] Server uses SSL? (yes/no) [no]: no // a little buggy with ssl.... on my computer.
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
/server add oftc <localhost or the server ip which runs znc >/10025
/set irc.server.oftc.username pericycle // the username same in the irc admin setting when you first set.
/set irc.server.oftc.password <my password, don't look, u dum as> // same password in znc admin.
/save
# do this step after you register your nickname.[1]
# /set irc.server.oftc.command "/msg nickserv identify xxx"

# now you have your own id, remember use register, else you id may be take up by other guys.
/msg NickServ register <xxx> <email@example.com>

# after doing this you can set the [1] operation in weechat
```

BTW, in libera chat, you may setup your sasl password after you have register your nickserv.

useful short cut:
alt + m enable mouse.
alt + up/down switch between channels.


reference:
- https://weechat.org/files/doc/weechat/stable/weechat_quickstart.en.html

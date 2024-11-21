# Start

Port uboot to a new soc riscv board.

## intro 


riscv is getting more and more popular...balabala...

See [jh7110 uboot upstream](https://patchwork.ozlabs.org/project/uboot/cover/20230329034224.26545-1-yanhong.wang@starfivetech.com/) as an example.
Reference: [Porting uboot to new board](https://bootlin.com/pub/conferences/2017/elce/schulz-how-to-support-new-board-u-boot-linux/schulz-how-to-support-new-board-u-boot-linux.pdf)

and work flow is as follow:

Brief look at uboot structure, only list the relative important stuff.
``` text

├── api
├── arch
│   ├── riscv/<cpu/dts>
├── board
├── cmd
│   ├── riscv
├── configs
├── drivers
│   ├── clk
│   ├── fastboot
│   ├── firmware
│   ├── gpio
│   ├── i2c
│   ├── net
│   ├── nvme
│   ├── pinctrl
│   ├── power
│   ├── pwm
│   ├── ram
│   ├── reset
│   ├── spi
│   ├── timer
│   ├── ufs
│   ├── usb
│   ├── video
│   ├── virtio
├── env
├── examples
├── fs
├── include
│   ├── configs
│   ├── dt-bindings
├── lib
├── Licenses
├── net
│   └── lwip
├── post
│   ├── cpu
│   ├── drivers
│   └── lib_powerpc
├── scripts
│   ├── basic
│   ├── coccinelle
│   ├── dtc
│   └── kconfig
├── test
│   
└── tools
```

## First
add your board soc file
location: `arch/riscv/<soc name>/` 
add these file:
- cpu.c 
- dram.c 
- Kconfig
- Makefile

e.g.
```text
---
 arch/riscv/cpu/jh7110/Kconfig                 |   28 +
 arch/riscv/cpu/jh7110/Makefile                |   10 +
 arch/riscv/cpu/jh7110/cpu.c                   |   23 +
 arch/riscv/cpu/jh7110/spl.c                   |   64 +
```

## Second
add your board specific file
location: `board/<vendor name>/` and
board header file
location: `include/<board name>.h`

add these file:
- board.c
- Makefile
- `<board name>.h`

```text
---
 board/starfive/visionfive2/Kconfig            |   53 +
 board/starfive/visionfive2/Makefile           |    7 +
 board/starfive/visionfive2/spl.c              |   87 +
 include/configs/starfive-visionfive2.h        |   49 +

```

## Third step

add the defconfig file and set your vendor, board, target, dts, and just implement dram and uart as the minimal system.

```
 arch/riscv/cpu/jh7110/dram.c                  |   38 +
 configs/starfive_visionfive2_12a_defconfig    |   79 +
 arch/riscv/dts/jh7110-u-boot.dtsi             |   99 +
 arch/riscv/dts/jh7110.dtsi                    |  573 +++++
 arch/riscv/dts/jh7110-starfive-visionfive-2-v1.2a.dts    |   12 +

```

your board Kconfig will be like this:
```Makefile
if TARGET_MY_BOARD
config SYS_BOARD
    default "my_board"
config SYS_VENDOR
    default "my_vendor"
config SYS_CONFIG_NAME
    default "my_board"
endif
```

e.g.

```Makefile
if TARGET_STARFIVE_VISIONFIVE2

config SYS_CPU
	default "jh7110"

config SYS_BOARD
	default "visionfive2"

config SYS_VENDOR
	default "starfive"

config SYS_CONFIG_NAME
	default "starfive-visionfive2"

endif
```

`SYS_CPU` let the build system check for path `<ARCH>/cpu/<SYS_CPU>/` for cpu specific target.
`SYS_VENDOR` and `SYS_BOARD` let the build system check for path `board/<SYS_VENDOR>/<SYS_BOARD>/` to build board specific target.

## Forth step
One/Two commit per IP 
e.g. 



list todo IPs part:
- Ram
- Ethernet
- Uart
- Spi
- Nand
- Emmc
- Clock 
- Gpio
- Reset
- Pmic

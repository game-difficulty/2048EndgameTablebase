.PHONY: all sharedlib install clean

all: sharedlib

sharedlib:
	meson setup --warnlevel 2 --werror --buildtype release builddir
	cd builddir && meson compile

install: sharedlib
	cd builddir && meson install

clean:
	rm -rf builddir debug

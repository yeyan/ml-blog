.PHONY: server dist hugo-server open-browser

server:
	make -j 2 hugo-server open-browser

dist:
	hugo -d dist -v -D

hugo-server:
	hugo server -D

open-browser:
	chromium http://localhost:1313

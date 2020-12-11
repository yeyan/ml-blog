# Personal Blog

This repo contains the source code for [Ye Yan's ML Blog](https://yeyan.github.io)


#### Local Development

Use the following command to start a webpack server:

```shell
npm start
```

Compiled static content will be generated under site/dist, which is a submodule pointing to
[ML Blog Repository](https://github.com/yeyan/yeyan.github.io).

Type the following command to update the blog hosted on GitHub:

```shell
# build the website
npm run build:hugo

# synchronize with github
cd site/dist
git add -A
git commit -m "Commit message"
git push
```

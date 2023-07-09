
- error: cannot pull with rebase: You have unstaged changes.
  - do git pull --rebase --autostash origin main
  - Git pull rebase is a method of combining your local unpublished changes with the latest published changes on your remote. Let's say you have a local copy of your project's main branch with unpublished changes, and that branch is one commit behind the origin/main branch.
  - https://stackoverflow.com/questions/23517464/error-cannot-pull-with-rebase-you-have-unstaged-changes

- Rename file with Git
  - [git rename file](https://stackoverflow.com/questions/6889119/rename-file-with-git)
  - Just adding for noobs like myself that using git mv automatically renames the file on your computer. I tried saving the file as a different name first before using git mv and was met with the error fatal: destination exists because of this silly mistake. – 


```
$ git clone git@github.com:username/reponame.git
$ cd reponame
$ git mv README README.md
$ git commit -m "renamed"
$ git push origin master
```
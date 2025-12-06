# Fork Setup Guide

This repository is a fork of the official [TiddlyWiki5](https://github.com/TiddlyWiki/TiddlyWiki5) repository. This guide helps you configure your local clone to track both the fork and the upstream repository.

## Setting Up the Upstream Remote

To keep your fork synchronized with the official TiddlyWiki5 repository, you need to add it as an upstream remote:

```bash
git remote add upstream https://github.com/TiddlyWiki/TiddlyWiki5.git
```

## Verify Remote Configuration

Check that both remotes are configured correctly:

```bash
git remote -v
```

You should see:
```
origin    https://github.com/selfapplied/TiddlyWiki5 (fetch)
origin    https://github.com/selfapplied/TiddlyWiki5 (push)
upstream  https://github.com/TiddlyWiki/TiddlyWiki5.git (fetch)
upstream  https://github.com/TiddlyWiki/TiddlyWiki5.git (push)
```

## Syncing with Upstream

### Fetch Latest Changes from Upstream

```bash
git fetch upstream
```

### Merge Upstream Changes into Your Branch

To update your main branch with upstream changes:

```bash
git checkout master
git merge upstream/master
```

Or to rebase:

```bash
git checkout master
git rebase upstream/master
```

### Push Updates to Your Fork

After syncing with upstream:

```bash
git push origin master
```

## Common Workflows

### Create a Feature Branch from Upstream

```bash
git fetch upstream
git checkout -b feature-name upstream/master
```

### Update a Feature Branch with Latest Upstream

```bash
git fetch upstream
git checkout feature-name
git rebase upstream/master
```

## Resources

- [Official TiddlyWiki5 Repository](https://github.com/TiddlyWiki/TiddlyWiki5)
- [Contributing Guidelines](./contributing.md)
- [GitHub's Fork Documentation](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks)

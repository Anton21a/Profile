#!/bin/bash
set -e

cd "$(dirname "$0")"

git add -A

if git diff --cached --quiet; then
  echo "Nothing new to commit."
else
  commit_message=${1:-"Update"}
  git commit -m "$commit_message"
fi

git pull --rebase origin main
git push origin main

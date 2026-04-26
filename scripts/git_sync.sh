#!/bin/bash
# Periodic git sync — commits any staged/unstaged changes and pushes to remote.
# Safe: won't commit if nothing changed. Won't force-push.
REPO=/home/shivamguptanit/RL-based-portfolio
cd $REPO

# Check for anything to commit
if git diff --quiet && git diff --cached --quiet && [ -z "$(git status --short)" ]; then
  echo "$(date) git_sync: nothing to commit" >> $REPO/artifacts/logs/git_sync.log
  exit 0
fi

# Stage all tracked + new files (exclude secrets/large binaries)
git add \
  artifacts/reports/ \
  artifacts/horizon_experiments/ \
  artifacts/run_history/ \
  src/ scripts/ config/ docs/ tests/ \
  NEXT_STEPS.md README.md \
  2>/dev/null || true

# Commit if staged
if ! git diff --cached --quiet; then
  TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  git commit -m "auto-sync: periodic checkpoint ${TIMESTAMP}

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>" >> $REPO/artifacts/logs/git_sync.log 2>&1
fi

# Push
git push origin $(git rev-parse --abbrev-ref HEAD) >> $REPO/artifacts/logs/git_sync.log 2>&1
echo "$(date) git_sync: pushed to origin/main" >> $REPO/artifacts/logs/git_sync.log

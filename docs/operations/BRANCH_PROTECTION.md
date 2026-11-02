# Production Branch Protection & Release Gate Specification

This document formalizes the mandatory branch protection rulesets, security scanning policies, and merge criteria governing the `main` branch of the AI Factory v4 repository.

---

## 1. Protected Branch Ruleset: `main`

The `main` branch represents deployable production artifacts. Direct pushes, force pushes, and unverified pull requests are cryptographically and administratively prohibited.

### 1.1 Pull Request Requirements
- **Require a pull request before merging**: Enabled.
- **Require approvals**: Minimum of `1` peer review approval required.
- **Dismiss stale pull request approvals when new commits are pushed**: Enabled.
- **Require review from Code Owners**: Enabled (governed by `.github/CODEOWNERS`).
  - Modifications to `/packages/auth/` require approval from `@factory-security-team`.
  - Modifications to `/manifests/` require approval from `@factory-core-team` and `@factory-ai-team`.
  - Modifications to `/deployments/` and `/infrastructure/` require approval from `@factory-infra-team`.
- **Require conversation resolution**: All review comments and discussions must be resolved before merging.

### 1.2 Status Checks & Release Gates
All of the following GitHub Actions status checks must report a green exit code before merge eligibility:
1. `Studio Web (Next.js 15 / React 19)`: Typecheck (`tsc --noEmit`), ESLint (`npm run lint`), and Next.js production build (`npm run build`).
2. `Backend CI (Python 3.11)`: Full workspace sync, manifest validation, anti-fake invariant guard, and test suite.
3. `Backend CI (Python 3.12)`: Cross-version Python runtime test suite.
4. `Secret & Credential Scanning`: Gitleaks and custom high-entropy secret scanner.
5. `Dependency & Lockfile Vulnerability Audit`: Lockfile consistency and audit.

### 1.3 Branch Integrity & History
- **Require branches to be up to date before merging**: Enabled (strict rebase or merge commit from `main`).
- **Require linear history**: Enabled (squash and merge or rebase merge enforced).
- **Require signed commits**: Enabled (GPG, SSH, or S/MIME signatures verified).
- **Do not allow bypassing the above settings**: Enabled for all contributors. Emergency bypass requires dual sign-off from designated release managers.
- **Allow force pushes**: Disabled.
- **Allow deletions**: Disabled.

---

## 2. Automated Scanning & Dependency Governance

### 2.1 Secret Scanning
- Continuous pre-receive and CI scanning via Gitleaks and AST pattern matching in `.github/workflows/security.yml`.
- Pinned secrets or test tokens are restricted to isolated test fixtures (`tests/` directory) and must never appear in runtime code or manifests.

### 2.2 Dependency Governance
- Dependabot configuration (`.github/dependabot.yml`) checks Python packages and NPM dependencies weekly.
- All dependencies must be locked via `uv.lock` (backend) and `package-lock.json` (frontend).
- Direct dependencies must specify explicit lower bounds; automated vulnerability alerts trigger immediate dependency patching.

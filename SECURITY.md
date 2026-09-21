# Security Policy

## Supported Versions

| Version       | Supported          |
| ------------- | ------------------ |
| 1.5.3         | :white_check_mark: |
| < 1.5.3       | :x:                |

## Scope

Joblib focuses on **parallelism, caching, and distributed computation**. Its
persistence layer is built on Python's `pickle` and is designed for **trusted
environments only** — see the
[persistence documentation](https://joblib.readthedocs.io/en/stable/persistence.html#security-considerations)
for the full security rationale and recommendations.

**Out of scope:**

- Any report whose exploit requires an attacker-controlled `.joblib` file
  (RCE via pickle, scanner bypass, DoS from crafted files, path traversal via
  pickle data). Joblib does not sandbox pickle deserialization — this is a
  Python language-level property, not a bug.
- Any report whose exploit requires the attacker to already control Python
  function attributes (e.g. `__module__`, `__qualname__`) or process
  environment variables (e.g. `LOKY_PICKLER`). Controlling these implies code
  execution by other means, so the reported issue is not an escalation.

**In scope:** issues exploitable through normal, trusted use of the public API
without a malicious file — e.g. path traversal or injection driven purely by
user-supplied function arguments to `Memory`.

## Reporting a Vulnerability

Please open a [GitHub security advisory](https://github.com/joblib/joblib/security/advisories/new)
or email `joblib-security@scikit-learn.org` (an alias to a subset of the
maintainers). Accepted reports will be patched privately before a dedicated
bugfix release.

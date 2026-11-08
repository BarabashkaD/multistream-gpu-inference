# Contributing

Thank you for helping. By opening a pull request you agree to the terms on this page.

## License of contributions

This project is licensed under the GNU Affero General Public License, version 3 or (at your option) any later
version, with the additional terms in [NOTICE](NOTICE). Your contribution is accepted under the same license.

## Relicensing grant

So that the project can later be offered under other terms, for example a dual license or a permissive license
once it no longer depends on AGPL components, each contributor grants the following for every contribution that is
merged:

> I grant Dmitry Veremeev a perpetual, worldwide, non-exclusive, royalty-free, irrevocable license, with the right
> to sublicense, to use, copy, modify, distribute and relicense my contribution under any license terms, including
> permissive and proprietary licenses. I confirm that I have the right to grant this license for my contribution.

You keep the copyright in your contribution.

## Developer Certificate of Origin

Sign off every commit with `git commit -s`. The sign-off certifies the
[Developer Certificate of Origin 1.1](https://developercertificate.org/): that you wrote the change or otherwise
have the right to submit it under the project's license.

## Pull requests

- Keep each change focused. If it affects performance, say what you measured, on which GPU, and how.
- Run the test suite on a machine with an NVIDIA GPU: `python -m pytest` runs the tests that need no GPU, then
  `python -m pytest -m gpu` runs the GPU tests. Name any tests you could not run.
- Every new source file starts with the license header used in the existing files, and every new module, class
  and function has a docstring.
- Tick the contributor terms in the pull request template.

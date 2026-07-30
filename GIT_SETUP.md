# Git commit guidance

Suggested sequence of commits when building the project locally and pushing to GitHub:

1. Initial repo and README
   - `git init`
   - `git add README.md`
   - `git commit -m "chore: init repo with README"`

2. Add project skeleton (server/client/model/utils, requirements)
   - `git add server.py client.py model.py utils.py requirements.txt .gitignore`
   - `git commit -m "feat: add FL server/client skeleton and dependencies"`

3. Implement core training logic and example model
   - `git add model.py utils.py client.py server.py`
   - `git commit -m "feat: implement CNN model, data loaders, and FL training loop"`

4. Add Docker support and scripts
   - `git add docker/ docker-compose.yml run_local.sh`
   - `git commit -m "chore: add Docker/Docker-Compose for server and clients"`

5. Polish README and add examples
   - `git add README.md`
   - `git commit -m "docs: improve README with quickstart and examples"`

6. Push to GitHub
   - `git remote add origin <your-git-url>`
   - `git branch -M main`
   - `git push -u origin main`

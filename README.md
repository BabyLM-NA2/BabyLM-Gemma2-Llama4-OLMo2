## Cloning a Repository

To begin contributing to a project, you'll first need to clone the repository to your local machine.

1. **Find the repository you want to clone.**
   - Go to the repository's page on GitHub (or other Git hosting platforms).
   - Click on the green **Code** button and copy the URL (HTTPS or SSH).

2. **Open your terminal or Git Bash.**
   
3. **Clone the repository using the following command:**

   ```bash
   git clone https://github.com/BabyLM-NA2/BabyLM2025.git
   ```

4. **Navigate into the cloned directory:**
   
   ```bash
   cd BabyLM2025
   ```

## Making Changes and Committing

Once you’ve cloned the repository, you can make changes to the code.

**VDO:** https://www.youtube.com/watch?v=nCKdihvneS0

1. **Create a new branch:**

   Before making any changes, it's good practice to create a new branch for your work:

   ```bash
   git checkout -b <branch-name>
   ```

   Replace `<branch-name>` with a descriptive name for your branch.

2. **Make your changes** to the code or documentation.

3. **Stage your changes:**

   ```bash
   git add .
   ```

   This stages all the changes you've made. You can also specify individual files instead of `.` to stage specific files.

4. **Commit your changes:**

   ```bash
   git commit -m "Description of changes"
   ```

   Be sure to write a clear and concise commit message describing the changes you made.

## Pushing Changes to GitHub

After committing your changes locally, you need to push them to the remote repository.

1. **Push your branch to GitHub:**

   ```bash
   git push origin <branch-name>
   ```

   Replace `<branch-name>` with the name of the branch you created earlier.

## Creating a Pull Request

Once your changes are pushed to GitHub, you can create a pull request (PR) to propose your changes to the original repository.

1. **Go to the repository page** on GitHub.

2. **Click the "Compare & pull request" button** next to your branch.

3. **Add a title and description** for your pull request, explaining what changes you made and why they should be merged.

4. **Click "Create pull request"** to submit your PR.
4. **Push the updates to your fork:**

   ```bash
   git push origin main
   ```

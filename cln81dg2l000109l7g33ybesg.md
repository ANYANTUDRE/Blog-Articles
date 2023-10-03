---
title: "Git fundamentals for Data Science and Machine Learning"
datePublished: Sun Oct 01 2023 22:31:16 GMT+0000 (Coordinated Universal Time)
cuid: cln81dg2l000109l7g33ybesg
slug: git-fundamentals-for-data-science-and-machine-learning
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1696191457550/de6d4179-9e29-4946-ab1c-4945fdf6f4a0.png
ogImage: https://cdn.hashnode.com/res/hashnode/image/upload/v1696199176287/0ff59f0d-6321-410e-82d9-29f2d25dbb21.png
tags: artificial-intelligence, github, data-science, machine-learning, git

---

A few months ago, I embarked on a journey to master Git. Looking back today, it wasn't a choice, but a necessity for my summer Machine Learning internship. If you aren't already aware of Git's indispensable role in software development and Data Science, this article is for you. Even if you've already used this tool, I'm confident you'll discover something new by the end of your reading.

So, enough with the preamble - let's dive into an exciting journey into the realm of version control with this remarkable tool and explore the reasons why every Data Scientist and Machine Learning Engineer should grasp the fundamentals of Git.

## **What is Git?**

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1696191528429/99685da4-2d84-40f7-8259-48c906996db5.png align="center")

A classic definition you’ll find about Git is that - ***it's a widely used distributed version control system in software development, designed for tracking and managing changes in computer files and source code over time.***

In simple terms, Git is a free and open-source tool utilized in large-scale projects for code collaboration. It helps maintain a comprehensive history of all modifications, facilitates seamless collaboration among developers, and ensures that project changes are well-organized, synchronized, secure, and easily accessible to teams of all sizes.

## **Don't confuse Git with GitHub: Git != GitHub**

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1696191834070/9b5c49de-0749-4c98-b4f1-e6a5a0cfb8a4.png align="center")

Like many beginners, I used to think Git and GitHub were the same, but actually, they're not.

Git serves as the core or the underlying technology, providing a command-line client (CLI) for tracking and merging source code changes. As already mentioned above, **it's a distributed version control system or software.**

In contrast, **GitHub is a web platform built on top of Git**, which enhances Git's capabilities and usability. Additionally, GitHub offers features like user management, pull requests, and automation. Alternatives like GitLab and SourceTree provide similar version control and collaboration features.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1696195628011/5e0ca31f-6fca-4d4e-90d2-898422909b1c.png align="center")

## **Top 5 reasons why Git is important in Data Science and ML**

As Data Science, Data Engineering and Machine Learning operations become ever more complex, Git is becoming one of the most integral and essential tools for practitioners in these areas. So, knowing Git is no longer an option and these are the reasons:

**1️⃣ | Version control and file recovery:**

Git tracks changes in code, data, and project files over time, enabling professionals to maintain and back up a history of their work, making it easy to revert to previous versions or states when needed.

**2️⃣ | Team Collaboration:**

For team-based Data Science projects, a version control system like Git allows team members to work and collaborate efficiently on the same project, at the same time, ensuring everything stays organized and safe.

**3️⃣ | Reproducibility:**

Git safeguards the reproducibility of data experiments by documenting the exact code and data versions used, facilitating result replication and validation.

**4️⃣ | Innovation through experimentation:**

Data professionals can create branches in Git to experiment with various data processing methods, models, or algorithms without affecting the main project, making it a valuable tool for testing and innovation.

**5️⃣ | Efficient workflow management:**

Git can be integrated into workflow automation and continuous integration/continuous deployment (CI/CD) pipelines. This helps automate testing, deployment, and model retraining, improving the efficiency of data science and engineering workflows.

## **Mastering basic Git commands & workflow**

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1696194058846/b4c41e3f-9e20-4033-90b6-b6c50d32265a.png align="center")

### **Git terminology**

* **Repository** – it’s like a "database" containing a whole project and its branches and commits.
    
* **Branch** − A repository's alternative state or route of development.
    
* **Merge** − Merging two (or more) branches into one branch.
    
* **Clone** − The process of locally copying a remote repository.
    
* **Origin** − Common alias for the remote repository from which the local clone was created.
    
* **Main/Master** − Common names for the root branch, which is the main repository.
    
* **Stage** − Choosing which files to include in the new commit at this stage.
    
* **Commit** − A saved snapshot of staged changes made to the file(s) in the repository.
    
* **Push** − Sending your changes to a remote repository for public viewing.
    
* **Pull** − getting everybody else's changes to your local or personal repository
    
* **Pull Request** − Mechanism to review and approve your changes before merging to main/master.
    

### **Top 10 most useful and used Git commands**

* **git init** − initialize a new Git repository on your local computer (in a folder).
    
* **git clone** – copy an already-existing Git repository from a remote server (like GitHub) to your computer so you can work on it.
    
* **git add** – this tells Git to start tracking changes in the file(s) (staging).
    
* **git status** - Show the files you have modified and what needs to be committed.
    
* **git commit** – save a snapshot (commit) of the chosen file(s) with a message explaining what you did.
    
* **git push** - Send your saved snapshots (commits) into the remote repository so others can see and use your work.
    
* **git pull** – Pull or simply get recent commits made by others from the remote repository into your local computer.
    
* **git branch** - It shows all the different "branches" of your project, which are like separate paths of development.
    
* **git checkout** − Switch branches or undo changes made to local file(s).
    
* **git merge** – to combine changes from one branch into another, usually used to bring new features into the main project.
    

I found this amazing and interesting gift on a [LinkedIn post](https://www.linkedin.com/posts/ginacostag_data-git-github-activity-7106275054933422080-oC-F?utm_source=share&utm_medium=member_desktop) that summarizes the workflow of Git and GitHub for three (3) people collaborating on the same project.

![Image Credit: Gina Acosta Gutiérrez](https://cdn.hashnode.com/res/hashnode/image/upload/v1696195882577/fba34a10-3a53-4870-bb27-2953c95c5079.gif align="center")

## **A fun fact about Git**

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1696197363964/997e3f22-8fe6-4963-a904-6eec20ab4841.png align="center")

Git's origin story is intriguing. It was initially developed by **Linus Torvalds**, the creator of the Linux operating system. He created Git in 2005 to help manage the development of the Linux kernel. What's interesting is that Torvalds originally named the tool ***"Git"*** as a playful reference to the slang term used in British English, meaning ***"an unpleasant or contemptible person."*** It was his way of expressing frustration with the existing version control systems available at the time.

Torvalds even quipped, “I'm an egotistical bastard, and I name all my projects after myself. First Linux, now Git.”

So, if you too are egotistical and unhappy with Git, you know now what you have to do :)

## **Some big companies & open-source projects that use Git**

### **Companies**

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1696195821287/fd395302-f792-4546-be2a-4705c2f14c4e.png align="center")

Many big tech companies rely on Git for managing the source code of their various software projects. It's the case of:

* Microsoft
    
* Apple
    
* Google
    
* Facebook
    
* Amazon
    

### **Open-Source Projects**

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1696212179757/f361f8a6-b3b6-4936-9bc2-bf2a85f212c4.png align="center")

Git is widely adopted across the open-source community for code management and collaborative development. Some open-source projects using Git include:

* Linux Kernel
    
* Kubernetes
    
* Docker
    
* Apache Software Foundation with Hadoop and Spark
    
* Mozilla Firefox
    
* Node.js
    

## **Free resources to learn Git and GitHub**

For beginner-friendly YouTube tutorials:

* **DataMites:** [Git Tutorial for Data Science - DataMites Training Institute](https://www.youtube.com/watch?v=IXpQQ37GGlc)
    
* **Programming with Mosh:** [Git Tutorial for Beginners: Learn Git in 1 Hour](https://www.youtube.com/watch?v=8JJ101D3knE)
    
* **FreeCodeCamp:** [Git and GitHub for Beginners - Crash Course](https://www.youtube.com/watch?v=RGOj5yH7evk)
    
* **Amigoscode:** [Git and GitHub Tutorial For Beginners | Full Course \[2021\] \[NEW\]](https://www.youtube.com/watch?v=3fUbBnN_H2c&t=26s)
    
* **Kevin Stratvert:** [Git and GitHub for Beginners Tutorial](https://www.youtube.com/watch?v=tRZGeaHPoaw&t=10s)
    

For advanced YouTube tutorials:

* **FreeCodeCamp:** [Git for Professionals Tutorial - Tools & Concepts for Mastering Version Control with Git](https://www.youtube.com/watch?v=Uszj_k0DGsg)
    
* **Tech With Tim:** [Intermediate GitHub Tutorial](https://www.youtube.com/watch?v=7h6_aZZ_iNg)
    

For free PDF tutorials and cheat sheets:

* **GitHub Education:** [Git cheat sheet](https://education.github.com/git-cheat-sheet-education.pdf)
    
* **KDnuggets:** [Git Cheatsheet](https://www.kdnuggets.com/publications/sheets/Git_Cheatsheet_KDnuggets.pdf)
    
* **Article on freeCodeCamp website:** [An introduction to Git: what it is, and how to use it](https://www.freecodecamp.org/news/what-is-git-and-how-to-use-it-c341b049ae61/)
    

## **References & Final Thoughts**

* **Valohai - Git for Data Science:** [What Every Data Scientist Should Know about Git](https://valohai.com/blog/git-for-data-science/)
    
* [Git official documentation](https://git-scm.com/docs)
    

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1696200421283/65d43e1a-5123-4003-8c81-084c15a29f2a.jpeg align="center")

In conclusion, being a Data Scientist doesn't require you to be a Git expert but you need to understand its fundamentals and know some basic commands. When I initially delved into Git, I faced frustration due to a lack of understanding of the workflow, despite my grasp of the terminology. This article has been written to help elucidate this workflow for others.

Therefore, it aims to provide an explanation of what Git is and underscore its importance in Data Science and Machine Learning workflow. I hope that you find this interesting and that you'll use Git in your day-to-day work and future Data Science projects.

**Thank you for reading, and please feel free to share your feedback or reach out to me with any further questions on** [**LinkedIn**](https://www.linkedin.com/in/anyantudre/)**.**

**See you in the next post!**
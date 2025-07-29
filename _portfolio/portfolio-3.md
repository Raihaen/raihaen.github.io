---
title: "Scheduling SaaS concept"
excerpt: "Idea I wanted to test at the Just build it hackathon"
collection: portfolio
---

The idea was basically an interface with a chat, a window to upload your files and a schedule calendar-like window. The user would load all the necessary files and ask the chatbot to create a schedule for him following specific constraints.

This was done by prompting Claude with the necessary info on how to turn the user's request into an operational research problem and generate a script to solve it and return a `.csv` file (using google's OR tools mainly). The ai can also ask the user about any missing info he encouters (making this work was a bit tricky).
Once the code is ready, it would be run in the background and shown to the user in a fancy way.

You can find the repo on my github (or comeback to this page in a few days and maybe I'll have added a link by then).

I worked on this project alongside Abhishev Thomas.
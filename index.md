---
layout: default
title: Home
seo:
  type: Course
  name: {{ site.title }}
nav_order: 1
---

# {{ site.tagline }}

<!--{% if site.announcements %}
{{ site.announcements.last }}
[Announcements](announcements.md){: .btn .btn-outline .fs-3 }
{% endif %}-->

The field of natural language processing (NLP) has been transformed by massive
pre-trained language models.  They form the basis of all state-of-the-art
systems across a wide range of tasks and have shown an impressive ability to
generate fluent text and perform few-shot learning.  At the same time, these
models are hard to understand and give rise to new ethical and scalability
challenges.  In this course, students will learn the fundamentals about the
modeling, theory, ethics, and systems aspects of large language models, as
well as gain hands-on experience working with them.

## Staff

{% assign instructors = site.staffers | sort: 'index' %}
{% for staffer in instructors %}
{{ staffer }}
{% endfor %}

## Coursework

Your grade is based on two activities:

1. In-Course Question Answering (50%)
2. Assignments (50%)

### In-Course Question Answering

In each class, you will be asked some questions. Every student can only answer every question for one time. The student who correctly answer the question for the first time will be granted 1 credit. 
The final scores will be calculated based on the accumulated credits in the whole semester.

### Assignments

There will be five assignments in the whole semester. Every assignment has a deadline, students should submit the completed assignment to a Google form before the deadline.

<!-- ## Logistics

**Where**: Class will by default be in person at
[200-002](https://goo.gl/maps/8ADRSg7nJ9xZC2Zd7) (History Corner).  The first
two weeks will be remote (in accordance with University policies);
[Zoom information](https://canvas.stanford.edu/courses/149841/external_tools/5384) is posted on Canvas.

**When**: Class is Mondays and Wednesdays 3:15-4:45pm PST.

**Links**:
- [Ed](https://canvas.stanford.edu/courses/149841/external_tools/24287?display=borderless):
  This is the main way that you and the teaching team should communicate:
  we will post all important announcements here, and you should ask
  all course-related questions here.
  For personal matters that you don't wish to put in a private Ed post, you can
  email the teaching staff at [cs324-win2122-staff@lists.stanford.edu](mailto:cs324-win2122-staff@lists.stanford.edu).
- [Canvas](https://canvas.stanford.edu/courses/149841): The course Canvas page
  contains links and resources only accessible to students ([Zoom link](https://canvas.stanford.edu/courses/149841/external_tools/5384) for
  remote classes).
- [Gradescope](https://www.gradescope.com/courses/342794): We use Gradescope
  for managing coursework (turning in, returning grades).  Please use your
  @stanford.edu email address to sign up for a Gradescope account.

**Video access disclaimer**: A portion of class activities will be given and
recorded in Zoom. For your convenience, you can access these recordings by
logging into the course Canvas site. These recordings might be reused in other
Stanford courses, viewed by other Stanford students, faculty, or staff, or used
for other education and research purposes. If you have questions, please
contact a member of the teaching team at [cs324-win2122-staff@lists.stanford.edu](mailto:cs324-win2122-staff@lists.stanford.edu).

## Class

Each class is divided into two parts:

1. **Lecture** (45 minutes): an instructor gives a standard lecture on a topic
   (see the [calendar](/calendar) for the list of topics).  Lectures are be
   based on [these lecture notes](/lectures).

1. **Discussion** (45 minutes): there is a student panel discussion on the
   required readings posted on the [calendar](/calendar).
 -->
# Provider Network Access and Cost Planner: Explained Simply

The big picture: the system is a new hire at a health plan's front desk. A member asks for help. The new hire understands the request, checks the rules, reads the paperwork, estimates how expensive the member will be, and finds them the right doctor.

- **AI** = the reasoning and planning side (finding, checking, assigning)
- **MI** = the learning side (predicting from data)

---

## Why AI and MI need each other

You don't need cost to find *a* doctor. You need it to find the **right** doctor when doctors are limited.

**1. Doctors fill up by workload, not headcount.**
A teacher with 25 students, 10 of whom need extra help every day, has a much harder class than one with 25 students who don't. A PCP with 1,500 healthy members is fine. A PCP with 1,500 very sick members is drowning. MI tells the AI planner who the "heavy" patients are, so no single doctor gets too many.

**2. When close doctors are scarce, the sickest go first.**
One open cardiologist 3 miles away, two members who need one. Member A is healthy. Member B has heart failure and two ER visits. B gets the close one, because a sick member forced to drive 20 miles skips care and ends up in the ER.

**3. Access drives cost.**
Members far from a doctor skip checkups. A problem a $150 visit would catch becomes a $15,000 hospital stay. So the AI's travel miles are one of the clues MI uses to predict cost.

**The loop:**

```
AI (A*) measures distance  ->  MI uses distance to predict cost
MI predicts cost           ->  AI (CSP) uses cost to assign doctors fairly
```

Real-world version: when a big provider group leaves the network, the network team asks "where do these members go?" and also "which doctors can absorb them, and who needs to be placed first?"

---

## Part 1: Find the route, estimate the cost

**In one line:** AI finds and assigns the doctor. MI estimates the member's cost in dollars.

### AI Part 1: Search + CSP

| Tool | Job | Analogy |
|---|---|---|
| A* search | Finds how far each doctor is (travel miles) | The GPS |
| CSP | Picks *which* doctor each member gets, following the rules (capacity, distance, specialty, not too many expensive patients per doctor) | The wedding seating chart |

- **A\*** scores each step as f = g + h: g is the miles already traveled, h is a guess of the miles left (count blocks, no diagonals). The guess never overestimates, so the first route found is the shortest.
- **CSP pieces:**
  - **Backtracking:** try a seat; at a dead end, undo and try another.
  - **MRV:** seat the pickiest guest first.
  - **Forward checking:** after each seat, cross off the seats that are no longer possible.
  - **Min-conflicts:** when a doctor leaves, fix only the people who lost their seat.

### What we actually built for AI Part 1 (Nov 9)

- **600 made-up members**, each with a home location, the type of doctor they need, and a stand-in high-cost yes/no (until MI Part 2 builds the real one).
- **Open seats:** only in-network doctors accepting new patients have seats, and the seats add up to 115% of what's needed, so it's tight on purpose. Each doctor can take high-cost members for at most 35% of their seats.
- **Each specialty is its own puzzle.** Cardiology seats have nothing to do with dermatology seats, so we solve them separately. Smaller puzzles are faster.

**The race between methods (same puzzles, same 200,000-step limit):**

| Method | Plain meaning | Solved (of 7) | Steps |
|---|---|---|---|
| Plain backtracking | Seat people in list order, undo when stuck | 5 | 416,759 |
| + Forward checking | After each seat, cross off full doctors for everyone else | 5 | 408,276 |
| MRV + forward checking | Also seat the pickiest member first | 6 | 200,237 |
| + LCV | Also try the doctor that blocks the fewest others | 6 | 200,237 |
| **+ Look-ahead** | **Before each seat, check that everyone still waiting can fit** | **7** | **480, no undoing** |

**Why the look-ahead was needed:** forward checking only notices when ONE member has no seat left. It can't see when a GROUP of members is fighting over too few seats, so the search walks into a dead end and keeps undoing. The look-ahead checks the whole group each time. Analogy: before seating a guest, the planner quickly counts whether the remaining guests still fit in the remaining chairs.

**Why no AC-3:** AC-3 checks rules between two guests at a time. "A doctor has 8 seats" involves everyone who could pick that doctor, so it isn't a two-guest rule.

**Results:**
- 465 of 600 members (77.5%) placed inside the federal distance limit, 4.88 miles on average, no doctor over capacity.
- 77 members have **no doctor close enough** (fix: recruit doctors there).
- 58 members have **close doctors that are full** (fix: more open seats).
- Only cardiology and endocrinology hit the 90% target. Primary care was worst (71%) because its limit is just 5 miles.

**Stress test: a big doctor group leaves.** Pinecrest Medical Group (15 doctors) leaves, and 73 members lose their doctor.

| Way to recover | Placed afterward | Unaffected members who lost their doctor | Unaffected members switched |
|---|---|---|---|
| Min-conflicts (fix only what broke) | 401 | 10 | 4 |
| Start over from scratch | 402 | 12 | 14 |

Almost the same coverage, but repair disturbs fewer people (14 vs 26). Every move is a member forced to change doctors.

**Honest notes:**
- **Both methods bump some unaffected members** to the waitlist. There aren't enough open seats near Pinecrest, so people who live closer win the seats.
- **Min-conflicts never fully settled:** it used all 3,000 of its allowed steps.

### MI Part 1: EDA + Regression

| Tool | Job | Analogy |
|---|---|---|
| EDA | Look at the data first to understand it | A mechanic popping the hood before fixing anything |
| Linear regression | Predicts the dollar amount ("this member will cost about $9,800 next quarter") | Drawing the best straight line through the data |
| MAE and RMSE | The report card: how far off the predictions were | Grading the guess. MAE = average miss in dollars; RMSE punishes big misses more |

**What the model looks at (the clues), like a car insurance quote:**

| Clue | Why it predicts cost |
|---|---|
| Age | Older members usually need more care |
| Chronic conditions | Ongoing visits and drugs |
| Prior cost | Expensive last year often means expensive next year |
| ER visits | Pricey, and a sign of unmanaged problems |
| Hospital admissions | Even more expensive |
| Travel miles to PCP | **Comes from AI's A\***. Far away means skipped checkups |
| Saw PCP recently | Checkups catch problems early |
| Plan type and county | Different benefits and local prices |

**No peeking rule (leakage):** every clue must be something we'd know *before* the quarter we're predicting.

**Note:** Part 1 gives a **dollar estimate**. The yes/no **risk flag** comes in Part 2.

### What we actually built for MI Part 1 (Nov 9)

**The data:** 1,500 made-up members followed month by month from January 2025 to September 2026, which comes to 31,563 rows. The first 600 are the same members from AI Part 1. "Miles to PCP" comes straight from AI Part 1's A* search.

**The audit (popping the hood):**

| Found | What we did |
|---|---|
| 63 rows loaded twice | Removed |
| 472 rows missing plan type | Labeled "Unknown" instead of throwing them away |
| 40 negative monthly costs | Kept, because they're refunds (claim reversals) |
| Middle member costs $472 a quarter, average is $1,967 | The top 10% of rows are 65% of all cost. A few people drive most of the spending |

**What the data showed:**
- More chronic conditions means more cost: about $900 a quarter with none, over $10,000 with 5 or more.
- **Distance matters for the far-away group.** Members 9+ miles from a doctor:
  - only 14% saw their doctor recently (vs 69% for members within 2 miles)
  - went to the ER about twice as often
  - cost about $2,700 vs $1,840
- **One number can hide a pattern.** The overall correlation between miles and cost is almost zero (0.03), because most members live close. The effect only shows up when you look at the far-away group by itself.

**The no-peeking split:**
- Train on January to September 2025, test on January to February 2026.
- October to December 2025 is skipped on purpose. A September row's answer covers October to December. Training past September would let the answers leak into the test months.

**The scorecard (average miss on the test months):**

| Model | Average miss | Plain meaning |
|---|---|---|
| Guess the average for everyone | $2,446 | The bar to beat |
| "Next quarter = last quarter" | $2,610 | Worse! One bad quarter usually isn't repeated |
| **Linear regression** | **$2,168** | **Beats both. This is the baseline future models must beat** |
| Log version | $1,834 | Closer for typical members, but misses the huge bills by more |

**Honest weak spots:** the straight line predicts below $0 for 126 rows, and it can't reach the giant bills. That's why Part 2 builds a yes/no high-cost flag and later Parts use models that can bend.

---

## Part 2: Sort things into buckets

**In one line:** AI sorts paperwork into types. MI sorts members into "high-cost" or "not high-cost." Both are **classification**: putting things into buckets.

### AI Part 2: Perception (reading paperwork)

**Why the project needs it:** members and doctors send forms, and each form type kicks off a different job.

| Form | What happens next |
|---|---|
| Referral | Triggers a doctor search (AI Part 1) |
| Claim form | Becomes cost data for MI |
| Provider enrollment form | Adds a new doctor to the directory |

Analogy: the mailroom sorter. If a claim lands in the referral pile, the member waits and the cost data goes missing.

| Tool | Job | Analogy |
|---|---|---|
| Flatten | Turns the image into one long list of numbers (each pixel = a number) | Unrolling a paint-by-numbers sheet into a single line |
| Softmax classifier | Scores each form type and turns scores into percentages ("85% claim form, 10% referral, 5% enrollment") | A judge's scorecard that must add up to 100% |
| Confusion matrix | Shows which form got mistaken for which | A mix-up chart: "referrals got filed as claims 6 times" |
| CNN | Looks at small patches to find lines, boxes, logos | A magnifying glass sliding across the page |
| Augmentation | Trains on crooked, blurry copies | Practicing reading bad handwriting so messy scans don't fool it |

### What we actually built for AI Part 2 (Nov 9)

**The data:** 1,500 made-up scanned documents, 500 of each type, each 64 x 64 pixels. Text is drawn as gray bars, so there are no real forms or people. Each type has its own layout:
- **Claim form:** a grid of boxes plus a line-item table
- **Referral:** a letterhead, paragraphs of text, and a signature
- **Enrollment form:** a title bar, checkbox sections, and an ID photo box

For each type: 280 to train on, 60 to decide when to stop training, 60 for the final test, and 100 **messy scans** (tilted, blurry, noisy, shadowed) that the model never trains on.

**How the classifier works (written from scratch):**
1. Flatten: each image becomes 4,096 numbers, one per pixel.
2. Each document type gets a score: a weighted sum of all the pixels.
3. Softmax turns the 3 scores into percentages that add up to 100%.
4. Training nudges the weights until the right answer gets the highest percentage. It stops early when it stops improving on the validation images.

**Results:**

| Model | Clean scans | Messy scans |
|---|---|---|
| Trained on clean images only | 100% | **78%** (mixed all 3 types up) |
| Trained with augmentation | 100% | **99.7%** (1 mistake out of 300) |

**Augmentation** = practicing on messy copies. Each training image got 2 tilted, blurry, noisy copies. Like practicing reading bad handwriting before the test.

**What the model looks at:** you can draw its weights as pictures. It learned *where* ink usually sits: the photo box and checkbox column for enrollment forms, the middle paragraphs for referrals, the box grid for claims. That's also why a tilted scan fools it. The ink moves. A CNN fixes that by sliding small pattern-finders across the page (built in MI Part 3).

**Routing (the agent part):**
- If the model is at least 70% sure, the document goes straight to its next step.
- If it's less sure, a person checks it.
- On the 300 messy scans: 99.3% were routed automatically with **zero wrong routes**, and 2 went to a person.

**Honest notes:**
- **100% on clean scans** happens because the synthetic layouts are very different from each other. Real forms would be harder.
- **99.7% on messy scans is optimistic.** The practice copies used the same kinds of damage as the messy test scans, so the model practiced on the test's kind of mess. Real scanners do things we didn't simulate (folds, stamps, faxes).
- **The 70% line** was picked on the same messy scans it's reported on. A fair version would pick it on separate scans first.

### MI Part 2: Classification Pipeline (the risk flag)

**Why the project needs it:** the AI planner needs a simple yes/no ("is this a heavy patient?") to spread them across doctors and decide who gets placed first.

| Tool | Job | Analogy |
|---|---|---|
| Pipeline | Cleans the data the same way every time: scales numbers, turns categories into 0/1 columns | An assembly line |
| Logistic regression | Gives a probability of being high-cost | A weighted checklist that adds up to a percentage |
| Decision tree | Asks yes/no questions to reach an answer ("Over 65? More than 2 ER visits?") | The game 20 Questions |
| Stratified CV | Tests the model 5 times, each with the same mix of high-cost members | 5 practice exams with the same difficulty mix |
| Confusion matrix | Right and wrong flags, side by side | Same chart as AI Part 2 |

**The accuracy trap:** if only 10% of members are high-cost, a lazy model that says "no" to everyone scores 90% and catches **nobody**. So we grade with:

| Metric | Plain question it answers |
|---|---|
| Precision | Of the members we flagged, how many really were high-cost? |
| Recall | Of the members who really were high-cost, how many did we catch? |
| ROC-AUC | Pick one high-cost and one normal member at random. How often does the model rank the high-cost one higher? |

### What we actually built for MI Part 2 (Nov 9)

**The question:** will this member be high-cost next quarter? "High-cost" = the top 10%, which works out to $3,904 or more next quarter. That line was set from the training months only.

**The no-cheating rules:**
- **Same split as MI Part 1:** train on Jan to Sep 2025, skip 3 months, test on Jan to Feb 2026.
- **Practice tests (cross-validation) keep each member in one group.** Otherwise the model could study a member's March row and get tested on their April row, which is almost the same thing. Like seeing the answer key from yesterday's quiz.

**The four models on the test months (305 members really became high-cost):**

| Model | Caught | Missed | False alarms | Plain meaning |
|---|---|---|---|---|
| Always say "no" | 0 | 305 | 0 | 90% accurate and totally useless (the accuracy trap) |
| Logistic regression | 21 | 284 | 8 | Learned that "no" is usually safe |
| **Logistic regression, balanced** | **204** | **101** | **785** | **The champion** |
| Decision tree, balanced | 220 | 85 | 1,053 | Catches 16 more, but 268 more false alarms |

**"Balanced"** = each missed high-cost member counts about 9 times more during training, because "no" is 9 times more common. Same skill, just not afraid to say yes. Recall went from 7% to 67%.

**Precision of 21%** means about 1 in 5 flagged members really becomes high-cost. For a $250 outreach call aimed at a $3,900+ quarter, that can still be worth it. The exact cut-off will come from costs in the capstone.

**What pushes risk up:** age, chronic conditions, last year's cost. **What pushes it down:** a recent PCP visit (the access link again).

**Hand-off to the AI planner:** the model scores the 600 AI Part 1 members and flags the riskiest 60 (10%). This list is ready to replace the placeholder flag in the planner. The planner does not read it yet; wiring it in is the next integration step. At the default 50% line it would have flagged 195, too many for the planner's per-doctor cap.

**How Part 2 AI and MI connect:**
- Same skill (sorting into buckets), and the same report card (the confusion matrix).
- AI reads the claim forms, and those claims become the data MI learns from.
- MI's high-cost flag goes straight to the AI planner (Part 1's CSP) to spread heavy patients and set placement order.

---

## Part 3: Understand language, and learn deeper patterns

### What we built for AI Part 3 (Nov 4)

**Job:** turn "Find a Spanish speaking cardiologist within 10 miles of 33024" into action.

- **The data:** 606 made-up member requests across 4 intents (find a provider, check network, change PCP, report an access problem), plus 36 requests I wrote by hand in new wording to test the model honestly.
- **Intent (what do they want?):** bag-of-words + logistic regression. Count the words, learn which words point to which intent.
- **Slots (the details):** pattern matching pulls out the ZIP, distance, time and doctor name. A word list maps "heart doctor" to cardiology.
- **Rulebook:** 22 "IF this AND that, THEN do this" rules fired like dominoes (forward chaining). Every answer comes with a receipt of which rules fired (a proof trace).
- **Bouncer:** messages like "ignore the rules" get blocked no matter what.

**Results:**

| Measure | Result |
|---|---|
| Intent on familiar wording | 99.5% |
| Intent on new wording | 83.3% (the model memorized the practice test) |
| Details pulled out correctly when the intent was right | 100% |
| Sneaky "ignore the rules" messages caught | 10 of 10, no false alarms |

**Fixes to the professor's starter code:** it graded itself on the same sentences it trained on, grabbed "33" out of ZIP codes as a time, crashed on unknown buildings, and only coded 1 of its 3 rules.

### What we built for MI Part 3 (Nov 9)

Three questions, three kinds of model.

**1. Does a neural network beat MI Part 2's champion on the high-cost flag? No, it ties.**
- **MLP** = layers of tiny calculators (128, then 64). ReLU only passes positive signals forward. Dropout benches 20% of the calculators at random during practice so the team doesn't lean on one star.
- **Result:** it ranks members exactly as well as logistic regression (0.755 vs 0.755) while using about 600 times more parameters. The pattern in this data is mostly a straight line (older + more conditions + more cost last year), so a model that can bend has nothing extra to find. **The simpler model stays champion.**
- **It learned everything in one pass.** After epoch 1, training error kept falling but validation got worse. That means it was memorizing the training members. Early stopping kept the epoch-1 version.

**2. Can a network that looks at shapes read messy scans better? Yes, when practice data is clean.**
- **CNN** = a magnifying glass that slides small 3 x 3 pattern-finders over the whole page, then shrinks the picture. A checkbox counts as a checkbox wherever it lands.

| Model | Messy scans |
|---|---|
| AI Part 2 softmax, clean practice only | 78% |
| **CNN, clean practice only** | **90%** |
| Softmax + augmentation | 99.7% |
| CNN + augmentation | 97% |

- Without augmentation, the CNN wins by 12 points. With augmentation, the simple softmax catches up and slightly wins, because these documents are simple.
- **Saliency map** = a heat map of which pixels changed the answer. On a correct claim form it lights up the grid of boxes. On a wrong one it's smeared across noise. The model guessed because the evidence was blurry.

**3. Can a simple probability model explain expensive stretches? Yes, and you can read every number.**
- **HMM** = like guessing the season (hidden) from today's temperature (what you see). Hidden: is the member in an expensive stretch? Seen: this month's bill size.
- Counting from the training months (+1 to every count so nothing is ever exactly zero):
  - A normal member has a 3% chance of starting an expensive stretch next month.
  - A member in a stretch has a 69% chance of staying in it.
  - In a stretch, 22% of months are $5,000+. Normally, almost never.
- **Viterbi** = the most likely hidden story behind a member's bills.
- It predicts worse (0.67) because it only sees bill sizes, but every number in it can be explained to a care manager.
- **Honest limit:** the "hidden" state is really calculated from last-3-month cost, which is how we could count the tables directly. A true HMM learns states it can't see (Baum-Welch).

**How sure is the model?**
- **MC Dropout** = ask the same question 20 times with different calculators benched. Answers that spread a lot mean "not sure." The most unsure cases were members with a giant bill last year and few other warning signs: was it a one-time event? Those should go to a person.
- **Calibration** = when the model says 60%, does it happen 60% of the time?
  - **Weighting the rare class makes the percentages lie.** The weighted models say about 60% when the real answer is about 15%, because we told them to fear misses.
  - The unweighted MLP's percentages are honest, but it barely flags anyone.
  - **Use the weighted model to rank members, not to quote percentages.**

**Ablations (remove one piece, see what breaks):**

| Removed | Effect |
|---|---|
| Dropout | Almost nothing |
| Half the calculators | Almost nothing (the big network was wasted) |
| **Class weights** | **Recall collapses from 65% to 4%.** The single most important choice |

**The mistakes, sorted:**
- 589 false alarms: older, with conditions, but stayed low (people overlap)
- 174 false alarms: expensive last year, quiet next quarter (one-time events)
- 55 had some cost history, but the model stayed under the 50% line (median 42%)
- 51 "came out of nowhere": nothing in the record pointed to it. No model can catch these.

**Fairness:**
- The flag catches 91% of high-cost seniors.
- It catches **0% of high-cost members aged 18 to 44**, and 25% on Medicaid.
- Part of that is real: seniors really are high-cost more often (19% vs 4.5%). But the gap in catching people is too big.
- **Fixes:** separate lines by age group, more signals than age, and a person reviewing young members with sudden ER visits.

**Responsible AI:** the flag is for outreach calls, **never for denying care**. Unsure cases go to a person. And since I wrote the data's rules myself, these results show the method, not real-world truth.

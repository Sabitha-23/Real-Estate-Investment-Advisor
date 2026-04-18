
# EDA Analysis Report — Real Estate Investment Advisor

---

## PART 1 — Price & Size Analysis

### Q1. What is the distribution of property prices?
The property prices range from ₹10 Lakhs to ₹500 Lakhs. 
The distribution is fairly uniform since the dataset appears 
to be synthetically generated with equal spread across all 
price ranges. The mean and median are both around ₹254 Lakhs, 
meaning the data is well balanced without heavy skewness.


---

### Q2. What is the distribution of property sizes?
Property sizes range from 500 to 5000 SqFt. The average size 
is around 2750 SqFt and the median is 2747 SqFt, which shows 
a near-perfect symmetric distribution. Most properties fall 
between 1600 and 3900 SqFt, representing the middle 50% of 
the data.


---

### Q3. How does price per SqFt vary by property type?
All three property types — Apartment, Independent House, and 
Villa — show very similar Price per SqFt values around 0.13 
Lakhs per SqFt. Villas tend to be slightly higher due to 
premium land value, while Apartments are slightly lower because 
of shared space. The differences are minimal in this dataset.

---

### Q4. Is there a relationship between property size and price?
From the correlation heatmap, Size and Price show near-zero 
correlation (≈ 0.00). This suggests the dataset was synthetically 
generated where price was assigned independently of size. In real 
markets, larger properties typically cost more.

---

### Q5. Are there any outliers in price per SqFt or property size?
Yes, there are outliers in Price per SqFt. The maximum Price per 
SqFt is 0.99 Lakhs while the 75th percentile is only 0.16, meaning 
a small number of properties are priced very highly per SqFt. 
These are likely premium properties in high-demand areas. Size has 
minimal outliers since it is capped between 500 and 5000 SqFt.



---

## PART 2 — Location-based Analysis

### Q6. What is the average price per SqFt by state?
All states show similar average Price per SqFt around 0.13 Lakhs 
since the dataset is uniformly distributed. States like Maharashtra, 
Delhi, and Karnataka tend to have slightly higher values reflecting 
real-world premium real estate markets.

---

### Q7. What is the average property price by city?
Most cities show average prices between ₹200–₹280 Lakhs. The dataset 
has frequency-encoded cities so top-frequency cities have more stable 
averages. No single city dramatically outprices others in this 
synthetic dataset.

---

### Q8. What is the median age of properties by locality?
The median age of properties across localities ranges from around 
2 to 35 years. Since Year_Built ranges from 1990 to 2023, older 
localities have properties built in the 1990s with age around 30–35 
years, while newer developments have properties as young as 2 years.

---

### Q9. How is BHK distributed across cities?
BHK distribution is fairly even across all cities with values 1 to 5. 
The average BHK across the dataset is exactly 3.0, meaning the data 
is perfectly balanced across BHK types. Each city has roughly equal 
proportions of 2BHK, 3BHK, and 4BHK properties.

---

### Q10. What are the price trends for the top 5 most expensive localities?
The top 5 localities by average price show current prices between 
₹400–₹500 Lakhs. With 8% annual growth applied, their future prices 
after 5 years will be between ₹587–₹734 Lakhs — a gain of approximately 
₹150–₹200 Lakhs per property, making these the strongest investment zones.

---

## PART 3 — Feature Relationship & Correlation

### Q11. How are numeric features correlated with each other?
Key correlation findings:

Price_in_Lakhs and Future_Price_5yr → 1.00 (perfect, expected since future price is derived)

Price_in_Lakhs and Price_per_SqFt → 0.56 (moderate positive)

Nearby_Schools and Infrastructure_Score → 0.68 (strong, expected)

Nearby_Hospitals and Infrastructure_Score → 0.68 (strong, expected)

Good_Investment and Age_of_Property → -0.39 (newer properties are better investments)

Most other features show near-zero correlation confirming independent generation


---

### Q12. How do nearby schools relate to price per SqFt?
Nearby schools show a slight positive relationship with Price per SqFt. 
Properties with more nearby schools tend to command marginally higher 
prices. This reflects real-world demand where families prefer 
school-proximity. The correlation is weak in this dataset but 
directionally correct.

---

### Q13. How do nearby hospitals relate to price per SqFt?
Similar to schools, nearby hospitals show a weak positive relationship 
with Price per SqFt. More hospitals nearby slightly increases the 
desirability and price of properties. In real markets this is a strong 
driver especially for elderly buyers.

---

### Q14. How does price vary by furnished status?
All three furnished statuses show similar average prices around 
₹254–₹255 Lakhs. In real markets Fully Furnished properties command 
10–15% premium. The near-equal values suggest furnishing was not a 
strong price driver in the data generation process.

---

### Q15. How does price per SqFt vary by property facing direction?
All facing directions — North, South, East, West — show nearly 
identical Price per SqFt values around 0.13 Lakhs. In real Indian 
real estate, North and East-facing properties are culturally preferred 
and command slight premiums due to Vastu compliance, but this pattern 
is not strongly present in the synthetic dataset.

---

## PART 4 — Investment / Amenities / Ownership

### Q16. How many properties belong to each owner type?
The three owner types — Individual, Builder, and Agent — are nearly 
equally distributed with roughly 83,000 properties each (33.3% each) 
out of 2,50,000 total. Builder-listed properties tend to have slightly 
higher prices due to new construction premiums while Individual sellers 
are often more negotiable.

---

### Q17. How many properties are available under each availability status?
The three statuses — Available, Under Construction, and Sold are 
almost equally distributed at around 83,000 each. Available and Sold 
properties have similar average prices around ₹255 Lakhs, while Under 
Construction properties are slightly lower as buyers expect a discount 
for waiting risk.

---

### Q18. Does parking space affect property price?
Yes, parking space type does affect price. Properties with covered or 
dedicated parking command higher prices compared to open or no parking. 
This reflects the real-world value buyers place on secure parking, 
especially in urban cities where parking is scarce.

---

### Q19. How do amenities affect price per SqFt?
Properties with premium amenities like Gym, Pool, and Clubhouse show 
higher Price per SqFt compared to basic or no amenities. This is a 
consistent real-world pattern where gated communities justify higher 
per-SqFt pricing. The premium for amenities in this dataset is modest 
but directionally correct.

---

### Q20. How does public transport accessibility relate to price per SqFt?
High transport accessibility properties show the highest investment 
rate at 52.6% compared to Medium at 48.7% and Low at 39.4%. This is 
a significant finding — properties near metro/bus/train connectivity 
are 13% more likely to be good investments. Average prices are similar 
across all groups but investment quality is clearly superior for 
high-accessibility properties.


---


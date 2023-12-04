# Predict-Customer-Personality-To-Boost-Marketing-Campaign
Predict Customer Personality To Boost Marketing Campaign By Using Machine Learning
by Mochamad Galuh Saputra

## Problems
A company can grow rapidly by understanding the personality behavior of its customers, enabling it to provide better services and benefits to potential loyal customers. By analyzing historical marketing campaign data to enhance performance and target the right customers for transactions on the company's platform, our focus from this data insight is to create a predictive clustering model to facilitate decision-making for the company.

## Objective

## Data Understanding

**![](https://lh7-us.googleusercontent.com/rcMlorAEPiphVV8vV-CSBbkFwWl59X9HBwxv6IiMeYEFEeTLasMhoaBK4N3_tSIAANj9BzVr9TdeedjY9HwpwoYFMZTVVuuvxqm-jFq2R6EgqX3Ug6dAf5CkFpAdmCS51dc-IOaFeWyrzQjobdfBP12opQ=s2048)**
  
From the existing features, 6 new features will be created, namely:

1.  Conversion Rate
2.  Age
3.  Age Group
4.  Total Kids
5.  Total Amount Spent
6.  Total Transaction

### Feature Engineering
Here is the source code used for feature engineering.
**![](https://lh7-us.googleusercontent.com/vb6ETbcQ2fk5CGlUAqWpGKue_RjxE66SEQ6dEqDiaDjmXTyNQZUbUWKGgaQI7_zXFPTWNqO1Q1jY3Iv7pc56aGXUOeUEO12_rNwxTZwClQZ8uCwZj_jbzfP71aptPt6Nx-dFMFtGcIeHd_auqarJvx16Hg=s2048)![](https://lh7-us.googleusercontent.com/DlFAZg5qM8GnbWEn1uuCeHxMSS1FgS8R7mG1KA5l5wJxu-3dRf3uueesxLWLXFHOQ0hL8WKao4po5HgGHvoyQt4M5CiLah-aSxPOnquL9DoS5yAliiB-RvmF8Zx_TaSRh_bQAhCCs_TTFS3NGS5j3zSatg=s2048)![](https://lh7-us.googleusercontent.com/gPJWZ3OzaZ5IoEofuItL18fTSCqfS8I95kjxAkST3PhXcBST3lmfz0YsYiJu_GPDM4L-iiIiwRYevRY5L-wthdk1X78sefU0hMC8wD12KHQFv2Rn9lKBFWaIP4BmXqMUj8ZX7MVoBz0O8jSBBmy0gHVf6w=s2048)**

## Exploratory Data Analysis
**![](https://lh7-us.googleusercontent.com/UWb7N0D-LxY4ju4RqtTdbmqL0C1_ujUxGdPhUf9pSQ5nP5K24jrUOu7yr4C0dbrrAmv2dSZMsWqY78IaZxPOyZDXngbI_jS_Tq20EQoi40iyLZpyuZeKbtT7KOPqNby54V4_HMzQyRpQBNFEOnORL3lwUA=s2048)**

The graph that has a correlation with the Conversion Rate is **Income** and **Total Amount Spent**.

**![](https://lh7-us.googleusercontent.com/mLFBkbvsENyykFnUU_eB8gJkXr2qBxgweAMwPmfU06M2Xcx8ThT8tcbiN5Qsda6Sy1ABCcVqnnWqTkT_v03wOwgegnzQE1Rk4fjvVTVslNC7Wy6Z09DB0ZtsY5nKFpYvF2uHYHSwoczGI61ih_pp8dfo5A=s2048)**

From the Heatmap graph, features that have a correlation with the Conversion Rate above 0.20 are **Income** and **Total_amount_spent**.

**![](https://lh7-us.googleusercontent.com/kftMsjSUiaMay4750fzht4YZuIRcVuHV2VmxWqFNfah9j98O1m8qFkxXA3DweRdAJIxsORLZvV2TSgL9S7170woVuRFn5ZSbmiStki_PGSRBMd4vR2X5adQjI1SQEKXitSTht_8YL7TXBcx0pttOzAMPKQ=s2048)**

The **Income** feature shows a linear relationship with the **Total_amount_spent**. This means that as the income increases, the total amount spent will also increase.


From the results of EDA, it can be seen that when outliers in Income are disregarded, the Conversion Rate shows a positive correlation with Income and Total Amount Spent. Additionally, Total Amount Spent is linear with Income. This implies that as the customer's income increases, their total spending also increases, accompanied by an increase in the conversion rate.

No other graphs indicate that the conversion rate is influenced by or influences other factors.

This can be used as a reference for business recommendations, suggesting that to improve the conversion rate, the focus can be directed towards customers with higher income.

## Data Pre-Processing
**
![](https://lh7-us.googleusercontent.com/DEWy-6946NjlQwFwbYxzwdhOJAh3iwKFN-smSK4QirF5ix74ook_LRY4zbQBuewP-_pEvOvtxObi-0FJgXRgj4et930hUYNCx5JjiFx2OUqLuA3WumpWhO65o_G4Mpo6hJgEz-HMjIMVn3oJ3s9nRaPAOg=s2048)![](https://lh7-us.googleusercontent.com/FFDHRr2bkwNxUROCrxmXngWLZ7-VHR5jE6NKUsCz_2lWlZtWpT7EEnND3TAxZM9601uTpKh6h1Xhq_6W9ImPhgyOPAzGlG3dqqdjnl2p63K-fRf0rGKgeix8r6Xpp1hF8o007eiaCViG1oo0xxAJ1uB2dw=s2048)**
- There are no duplicate data. 
- There are NULL values in the Income feature.

**![](https://lh7-us.googleusercontent.com/oU2bDXvWXg5AwEiHhmq_RH5Hk92OJA-aZdL62H-kd1Ac5G_UpMEHUoPtsYVvPWSmXjnLneoUZ1vdGCjuU5IKqV-GKBKCmVRX4BvNgZXAtz666Wgad9FXxvbdziCRJDFAKTPYWYGMHg5XA6pwQt_cepL6eQ=s2048)![](https://lh7-us.googleusercontent.com/iXzwhY00Vrc9bNlEOzKv2bqsFK14X2X0o1Z5nZ2IES6o8hea7CByqJuKFyw5ygBTfv3G6leFLtGIdTeRDRVtqInqOL-krtNvdSyOqHkaxb3EprjWzC3-m0QawqNTbmojGyjHzbM29Qi0Ku21gM0fK07mAQ=s2048)
![](https://lh7-us.googleusercontent.com/3erILKwMdvoIhUVRAw93BtHvJuRmfSIzX5Jrqz6VPiJna5kazndYVxPxZ7G4vjPOqEwm9AlmPW4HXfHchizPM2_Dpr3jAnn7opmj7nf4zNlxta19bAwjkQg2_eJjm6EWRVbqyHtbEsXWVMvKWoD1-7nnmQ=s2048)
**
- Changing the values of the Marital Status feature to 2 values.
- Performing label encoding on the education feature.
- Performing one-hot encoding on the marital status and age group features.

## Standarization
Performing standardization using StandardScaler
**![](https://lh7-us.googleusercontent.com/9DuaZOn6d4FrwJmwM2hfb4a9jJrcvbkubFwqcmEHwzQ09tQ6w8AaaGyq4oYY7ih_U3ZBOXrCnTJQfsCjQRnCEjTk5hPQNJ77HUa8j46sn6nrfvFDhKKW9zBMnAB3VSXQXDKKHXieiwGwDF_CtEplQHaLtQ=s2048)![](https://lh7-us.googleusercontent.com/wtrDKddkVEsxvLV_Ul1DTr25DyrxWwbzUwxfDfEIGZAwXd8245S7sL7ljkSsGXYOiot696bM8l5Qlg3lugHG2i4-taq-z7gbciK9xjaiih3lWig6gze1mXXICkCpRDtNhDRmRAI5HYPG2iOWCCJR6oXT9g=s2048)**
## Modeling
**![](https://lh7-us.googleusercontent.com/gNpIDWQwLMjrKvv3Eio2b95JS3axE1vfghgxaMdJPZ2LqUZzTSHlIrNgu9d6zr8dst78ilENo1zDb90NDGoO9-1LqDl59Krd_jtmLk3c7S-Vm7Sb6dDwxip5rjv_cVErLVcy3HhpJVVuWwO85hKtAMm91A=s2048)![](https://lh7-us.googleusercontent.com/UaDCnSqaOGQQWjVACdK2K8fkMuKkZSiqvzXS_CrFvqfuA3zr3fPx9TPcyAjLpie_PsuOH4VvJwGqiEVPD4H3fu0gukQGAa2uibMEDFbGx-mjvBbunF-6EZEFRVGQaUuDThCZuVlmYDTXvOv4qgtGBAI6hA=s2048)**

It appears that the optimal number of clusters is 4 for clustering.

**![](https://lh7-us.googleusercontent.com/wKGm7DCz2eECHVgKFoKV1wyRQ10HQ3r_3Egkb_HP1-PuaItNs0QWRXHl1EjfTqmnNpDWhGWwzz4PfInuoxHRaK4GW0yj92GYdBsOB14Qk2yTUl5XzsOK3WU6xV_sZqgPTAfTjPoyv8mYMrD4v8hN2lwhgA=s2048)**
- High-Value Customer: This group has an average 'Recency' of around 24-25 days, with an average number of transactions of about 8-9 and an average total spending of approximately 127 million to 136 million. This group consists of 594 customers, which is approximately 27.19% of the total customers.

- Low-Value Customer: This group has a relatively higher 'Recency,' with an average of about 74-76 days, an average number of transactions of about 9-10, and an average total spending of about 136 million. This group comprises 568 customers, which is approximately 26.00% of the total customers.

- Medium-Value Customer: This group has relatively low 'Recency,' with an average of about 22-23 days, an average number of transactions of about 21-22, and an average total spending of about 1.106 billion. This group has 479 customers, which is approximately 21.92% of the total customers.

- Very Low-Value Customer: This group has a high 'Recency,' with an average of about 72-73 days, an average number of transactions of about 21-22, and an average total spending of about 1.151 billion to 1.151 billion. This group consists of 544 customers, which is approximately 24.90% of the total customers.


## Business Recommendation
- High-Value Customer: This group consists of a small portion of the total customers but contributes significantly to the total spending. Despite receiving a relatively small number of campaigns, the significant total spending underscores the importance of retaining and enhancing engagement with this group.

- Medium-Value Customer: Although their numbers are fewer, this group has a sufficiently high total spending, indicating significant potential for further growth. Appropriate marketing strategies, including special offers and increased interactions, can help boost engagement and loyalty among customers in this group.

- Low-Value Customer: Despite their contribution being not as substantial as other groups, the total spending from this group is still significant. Efforts to retain and increase the number of customers, along with enhancing the transaction value per customer, can contribute to increasing revenue from this group.

- Very Low-Value Customer: Although the number of customers in this group is quite large, their collective total spending is highly significant. Through marketing strategies focused on enhancing engagement and the transaction value per customer, the potential to increase contributions from this group is substantial.

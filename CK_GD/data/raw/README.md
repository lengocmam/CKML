# Đặt file CSV training data vào đây

## Format yêu cầu:

File: `heart_disease_data.csv`

### Cấu trúc:
- **23 cột**: 1 target + 22 features
- **Header row**: Tên các cột
- **Data rows**: Dữ liệu đã được chuẩn hóa

### Thứ tự cột:

```
HeartDisease, BMI, PhysicalHealth, MentalHealth, SleepTime,
Race_American Indian/Alaskan Native, Race_Asian, Race_Black, 
Race_Hispanic, Race_Other, Race_White,
Smoking, AlcoholDrinking, Stroke, DiffWalking, Sex, AgeCategory,
Diabetic, PhysicalActivity, GenHealth, Asthma, KidneyDisease, SkinCancer
```

### Ví dụ:

```csv
HeartDisease,BMI,PhysicalHealth,MentalHealth,SleepTime,Race_American Indian/Alaskan Native,Race_Asian,Race_Black,Race_Hispanic,Race_Other,Race_White,Smoking,AlcoholDrinking,Stroke,DiffWalking,Sex,AgeCategory,Diabetic,PhysicalActivity,GenHealth,Asthma,KidneyDisease,SkinCancer
0,-1.845,-0.047,3.281,-1.460,0,0,0,0,0,1,1,0,0,0,0,7,3,1,3,1,0,1
1,-1.256,-0.424,-0.490,-0.068,0,0,0,0,0,1,0,0,1,0,0,12,0,1,3,0,0,0
...
```

### Ghi chú:
- **HeartDisease**: 0 = Không bệnh, 1 = Có bệnh
- **Continuous features** (BMI, PhysicalHealth, MentalHealth, SleepTime): Đã standardized
- **Race features**: One-hot encoded (0 hoặc 1)
- **Binary features**: 0 hoặc 1
- **Ordinal features** (AgeCategory, Diabetic, GenHealth): Số nguyên

---

**Sau khi đặt file CSV vào đây, chạy:**

```bash
cd scripts
python train_models.py
```

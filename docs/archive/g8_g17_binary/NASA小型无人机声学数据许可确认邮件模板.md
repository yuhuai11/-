# NASA Small UAS Flyover Acoustics Data 许可确认邮件模板

## 收件人

```text
nikolas.s.zawodny@nasa.gov
```

## 主题

```text
Reuse permission clarification for Small UAS Flyover Acoustics Data (6s8fb29q)
```

## 正文

```text
Dear Dr. Zawodny,

I am conducting non-commercial academic research on acoustic UAV detection and
would like to use the “Small UAS Flyover Acoustics Data” dataset (NASA dataset
identifier 6s8fb29q) to train and evaluate a machine-learning model.

The NASA Open Data Portal lists the dataset with public access, but its dataset
page currently says that the license is not specified. Could you please confirm
whether the downloaded audio/MAT data may be:

1. processed into audio segments and features;
2. used for non-commercial machine-learning model training and validation;
3. described in an academic paper, with the dataset and NASA acknowledged; and
4. used to publish derived aggregate results and trained model parameters?

We will not redistribute the original recordings unless that is separately
permitted, and we will preserve the dataset provenance and provide the requested
citation/acknowledgment.

If a specific license or terms-of-use page applies to this dataset, a link to it
would be greatly appreciated.

Sincerely,
[Name]
[Institution]
[Contact information]
```

## 工程处理规则

- 收到明确书面回复前，`training_allowed`保持`false`；
- 保存完整邮件（含发件人、收件人和日期）；
- 将回复导出为PDF或EML并计算SHA256；
- 只依据明确答复更新许可证证据，不依据“公开访问”自行推断训练授权；
- 如果回复禁止训练或长期未回复，NASA数据退出G14训练来源，改用许可明确的替代源。

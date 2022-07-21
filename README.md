# Определение элементов дорожной инфраструктуры на железной дороге
Автор: Вячеслав Барков

![Элементы дорожной инфраструктуры](images/infrastructure.png)


## Цель работы
Разработка алгоритма определения железнодорожной колеи и подвижного состава для предотвращения чрезвычайных ситуаций на железной дороге. Содержит в себе несколько первостепенных задач определения элементов дорожной инфраструктуры: колеи (рельсошпальной решетки) и подвижного состава (локомотивы, грузовые вагоны, пассажирские вагоны).

## Решение
Решена задача семантической сегментации с использованием ансамбля из моделей с декодером PAN (Pyramid Attention Network) и энкодерами EfficientNet B4, NFNet (Normalization Free Net) с ECA (Efficient Channel Attention) и функцией активации SiLU, а также SE-ResNet. </br>
Обучение производилось при помощи Google Colab с использованием видеокарты Tesla P100.

## Результаты
[Railway_Infrastructure_Segmentation](Railway_Infrastructure_Segmentation.ipynb) - ноутбук с исследованием, обучением и предсказанием. Также ноутбук доступен на [Google Colab](https://colab.research.google.com/drive/1ZpusX5_8jv6mvNSe9pwRF0kU9XDCCETN) с дополнительной подготовкой окружения специально для Google Colab.</br>
Веса обученных моделей и история их обучения находятся в разделе [Github Releases](https://github.com/slavabarkov/railway-infrastructure-segmentation/releases) а также доступны на [Google Drive](https://drive.google.com/drive/folders/1NgPEgN2azZ_I7hPiESbTdO0O2xhYjfyd).


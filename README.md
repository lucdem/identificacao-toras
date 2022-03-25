# Detecção e determinação da idade de toras de madeira

# Identificação - Métodos de Visão Computacional Clássica

	A abordagem por métodos clássicos utilizada se baseia na transformada de Hough para círculos e distância de Mahalanobis. Numa primeira etapa é aplicado um blur intenso na imagem sendo então aplicado o algoritmo de detecção de bordas de Canny seguido da transformada de Hough.
	
	Em seguida seleciona-se os dois círculos com maior valor de acumulador no espaço de Hough, calculado o valor médio de seus pixels, sendo então calculada a distância de Mahalanobis de todos os pixels da imagem em relação a este valor médio.
	
	Feito isso, a transformada de Hough é utilizada novamente, desta vez com um blur menos intenso. Os círculos encontrados nesta etapa são avaliados com base nas distâncias de Mahalanobis, aqueles que possuírem muitos pixels com distância acima de um certo limiar são considerados falsos positivos e eliminados do resultado final.

# Identificação - Rede Neural

	Na abordagem por redes neurais foi utilizada segmentação de instâncias baseada na arquitetura ResNet50-FPN disponibilizada na biblioteca Detectron2.
  
# Determinação de Idade

	Para esta etapa foram utilizados apenas métodos clássicos, se baseando na detecção dos anéis de crescimento da árvore. A partir dos métodos de identificação descritos, a parte da imagem referente a base da tora é convertida para uma representação polar em preto em branco. Nela aplica-se uma equalização de histograma e o qual os valores acima e abaixo de um limiar são convertidos para 1 e 0, efetivamente classificando as regiões da imagem em claras ou escuras.
	Nesta representação binária da imagem se aplica fechamento e abertura, em seguida são percorridas 30 linhas diferentes indo do centro da representação polar para o perímetro, sendo contados o número de vezes que o valor muda nessa linha. Esse valor é dividido pela metade para obter o número de anéis encontrados, e adota-se a mediana dos valores obtidos nas 30 linhas como valor estimado para a idade da árvore no momento do corte.

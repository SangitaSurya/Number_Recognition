import pygame
import sys
import numpy as np
import cv2

# Initialize Pygame
pygame.init()

# Constants
WINDOWSIZE = (800, 600)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BOUNDRYINC = 10
IMAGESAVE = True
PREDICT = True

# Create the display surface
DISPLAYSURF = pygame.display.set_mode(WINDOWSIZE)
pygame.display.set_caption('Draw and Predict')

# Initialize variables
iswriting = False
number_xcord = []
number_ycord = []
MODEL = 'c:\\Users\\DELL\\Downloads\\bestmodel.h5'  # Replace with your model
LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]  # Replace with your labels
FONT = pygame.font.Font(None, 36)

image_cnt = 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)

            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == pygame.MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == pygame.MOUSEBUTTONUP:
            iswriting = False
            if len(number_xcord) > 1 and len(number_ycord) > 1:
                number_xcord = sorted(number_xcord)
                number_ycord = sorted(number_ycord)

                rect_min_x = max(number_xcord[0] - BOUNDRYINC, 0)
                rect_max_x = min(WINDOWSIZE[0], number_xcord[-1] + BOUNDRYINC)
                rect_min_y = max(number_ycord[0] - BOUNDRYINC, 0)
                rect_max_y = min(WINDOWSIZE[1], number_ycord[-1] + BOUNDRYINC)

                number_xcord = []
                number_ycord = []

                img_arr = np.array(pygame.surfarray.array3d(DISPLAYSURF))
                cropped_img = img_arr[rect_min_y:rect_max_y, rect_min_x:rect_max_x]

                if IMAGESAVE:
                    cv2.imwrite(f"image_{image_cnt}.png", cropped_img)
                    image_cnt += 1

                if PREDICT:
                    image = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
                    image = cv2.resize(image, (28, 28))
                    image = np.pad(image, ((10, 10), (10, 10)), 'constant', constant_values=0)
                    image = image / 255.0  # Normalize image to [0, 1]

                    label = LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))]

                    textSurface = FONT.render(label, True, WHITE, BLACK)
                    textRecobj = textSurface.get_rect()
                    textRecobj.left, textRecobj.top = rect_min_x, rect_min_y

                    DISPLAYSURF.blit(textSurface, textRecobj)

        if event.type == pygame.KEYDOWN:
            if event.unicode == "n":
                DISPLAYSURF.fill(BLACK)

    pygame.display.update()

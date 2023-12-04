import { Carousel } from "@mantine/carousel";
import { Center, Image, Stack, Text } from "@mantine/core";

export interface Props {
  imgURLs: string[];
  onSelect: (imgURL: string) => void;
}

export function ImageSelector({ imgURLs, onSelect }: Props) {
  return (
    <Stack
      spacing={10}
      sx={{
        borderRadius: "15px",
        padding: "10px",
        boxShadow: "0px 0px 10px rgba(0, 0, 0, 0.1)",
        border: "1px solid #eaeaea",
      }}
    >
      <Center>
        <Text
          color="#333333"
          weight={700}
          sx={{
            fontFamily: "Pretendard",
          }}
        >
          마음에 드는 배경 이미지를 선택해주세요.
        </Text>
      </Center>
      <Carousel
        withControls={false}
        dragFree
        slideGap={10}
        slideSize="30%"
        align="start"
      >
        {imgURLs.map((imgURL, index) => (
          <Carousel.Slide key={index} sx={{ width: 0 }}>
            <Image
              src={imgURL}
              onClick={() => onSelect(imgURL)}
              alt=""
              height={110}
              width={180}
              fit="cover"
              sx={{
                cursor: "pointer",
                ":hover": {
                  opacity: 0.9,
                },
              }}
              radius={10}
            />
          </Carousel.Slide>
        ))}
      </Carousel>
    </Stack>
  );
}

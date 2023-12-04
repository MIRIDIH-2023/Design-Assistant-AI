import { DialogBox } from "../components/DialogBox";
import { Image, MasterSlideProps, Slide, SlideProps, Text } from "react-pptx";
import { Affix, Button, rem } from "@mantine/core";
import { IconDownload } from "@tabler/icons-react";
import { Presentation } from "../components/Presentation";
import { useState } from "react";
import { predictLayout } from "../utils/api";
import { convertLayoutToComponent } from "../utils/converter";
import { ChildElement } from "react-pptx/dist/util";

function Generator() {
  const [downloadPPTX, setDownloadPPTX] = useState<() => Promise<void>>(
    async () => {}
  );
  const [slideList, setSlideList] = useState<
    ChildElement<SlideProps | MasterSlideProps>[]
  >([
    // <Slide key={1}>
    //   <Text
    //     style={{
    //       x: 1,
    //       y: 1,
    //       w: 4,
    //       h: 0.5,
    //       fontSize: 32,
    //       bold: true,
    //     }}
    //   >
    //     Your AI assistant
    //   </Text>
    //   <Text
    //     style={{
    //       x: 1,
    //       y: 1.6,
    //       w: 6,
    //       h: 0.5,
    //       fontSize: 32,
    //       bold: true,
    //     }}
    //   >
    //     for presenting ideas
    //   </Text>
    //   <Text
    //     style={{
    //       x: 1,
    //       y: 2.3,
    //       w: 6,
    //       h: 0.5,
    //       fontSize: 16,
    //     }}
    //   >
    //     Design Cat과 함께
    //   </Text>
    //   <Text
    //     style={{
    //       x: 1,
    //       y: 2.6,
    //       w: 6,
    //       h: 0.5,
    //       fontSize: 16,
    //     }}
    //   >
    //     내용을 구상하고
    //   </Text>
    //   <Text
    //     style={{
    //       x: 1,
    //       y: 2.9,
    //       w: 6,
    //       h: 0.5,
    //       fontSize: 16,
    //     }}
    //   >
    //     알맞은 배경과 레이아웃으로 디자인하세요
    //   </Text>
    //   <Image
    //     src={{
    //       kind: "path",
    //       path: "https://i.ibb.co/yFH8G05/Kakao-Talk-20230913-181150882.png",
    //     }}
    //     style={{
    //       x: 5.5,
    //       y: 1,
    //       w: "38%",
    //       h: "60%",
    //     }}
    //   />
    //   <Text
    //     style={{
    //       x: 1,
    //       y: 3.8,
    //       w: 6,
    //       h: 0.5,
    //       fontSize: 16,
    //       bold: true,
    //     }}
    //   >
    //     지금 바로 {'"'}~를 만들어줘.{'"'} 라고 요청해보세요!
    //   </Text>
    // </Slide>,
  ]);

  return (
    <>
      <Affix position={{ bottom: rem(30), right: rem(30) }}>
        <Button
          sx={{
            boxShadow: "0px 0px 10px rgba(0, 0, 0, 0.1)",
          }}
          leftIcon={<IconDownload />}
          radius="xl"
          onClick={() => downloadPPTX()}
        >
          PPTX 다운로드
        </Button>
      </Affix>
      <Presentation setDownloadPPTX={setDownloadPPTX}>{slideList}</Presentation>
      <DialogBox
        sx={{
          width: 700,
          position: "absolute",
          left: "50%",
          bottom: rem(20),
          transform: "translateX(-50%)",
        }}
        onGenerate={(text, imgUrl) => {
          const textList = text.split("\n");
          predictLayout(textList, imgUrl).then((layout) => {
            let textComponentList: JSX.Element[] = [];
            for (let i = 0; i < layout.length; i++) {
              const textComponent = convertLayoutToComponent(
                layout[i],
                textList[i],
                i === 0 ? imgUrl : undefined
              );
              textComponentList.push(textComponent);
            }
            setSlideList([
              ...slideList,
              <Slide key={slideList.length}>{textComponentList}</Slide>,
            ]);
          });
        }}
      />
    </>
  );
}

export default Generator;

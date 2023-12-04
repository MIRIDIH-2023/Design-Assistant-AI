import { Carousel } from "@mantine/carousel";
import Preview from "react-pptx/dist/preview/Preview";
import {
  MasterSlideProps,
  Presentation as ReactPPTXPresentation,
  SlideProps,
  render,
} from "react-pptx";
import { ChildElement } from "react-pptx/dist/util";
import { Dispatch, SetStateAction, useEffect } from "react";

export interface Props {
  children?: ChildElement<SlideProps | MasterSlideProps>[];
  setDownloadPPTX: Dispatch<SetStateAction<() => Promise<void>>>;
}

export function Presentation({ children, setDownloadPPTX }: Props) {
  useEffect(() => {
    setDownloadPPTX(() => async () => {
      render(<ReactPPTXPresentation>{children}</ReactPPTXPresentation>, {
        outputType: "blob",
      }).then((pptx) => {
        const blobURL = URL.createObjectURL(pptx as Blob);
        const tempAnchor = document.createElement("a");
        document.body.appendChild(tempAnchor);
        tempAnchor.href = blobURL;
        tempAnchor.target = "_blank";
        tempAnchor.download = "presentation.pptx";
        tempAnchor.click();
        document.body.removeChild(tempAnchor);
        URL.revokeObjectURL(blobURL);
      });
    });
  }, [children, setDownloadPPTX]);

  return (
    <Carousel
      mx="auto"
      maw={1150}
      withKeyboardEvents
      controlSize={40}
      styles={{
        control: {
          opacity: children?.length ? 1 : 0,
        },
      }}
    >
      {children?.map((slide, index) => (
        <Carousel.Slide key={index}>
          <Preview
            slideStyle={{
              backgroundColor: "white",
              width: "960px",
              height: "540px",
              margin: "0 auto",
              border: "1px solid rgba(0, 0, 0, 0.1)",
            }}
          >
            <ReactPPTXPresentation>{slide}</ReactPPTXPresentation>
          </Preview>
        </Carousel.Slide>
      ))}
    </Carousel>
  );
}
